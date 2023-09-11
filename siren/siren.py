import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from scipy.io import savemat
from torch.utils.checkpoint import checkpoint

class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()
    def forward(self, x, freq=25.):
        return torch.sin(freq * x)

def sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def fmm_modulate_linear(x, weight, styles, scale=1.0, noise=None, activation="demod"):
    """
    x: [batch_size, c_in, N]
    weight: [c_out, c_in]
    style: [batch_size, num_mod_params]
    noise: Optional[batch_size, 1, height, width]
    """
    batch_size, N, c_in = x.shape
    c_out, c_in = weight.shape
    rank = styles.shape[1] // (c_in + c_out)

    assert styles.shape[1] % (c_in + c_out) == 0

    # Now, we need to construct a [c_out, c_in] matrix
    left_matrix = styles[:, :c_out * rank] # [batch_size, left_matrix_size]
    right_matrix = styles[:, c_out * rank:] # [batch_size, right_matrix_size]

    left_matrix = left_matrix.view(batch_size, c_out, rank) # [batch_size, c_out, rank]
    right_matrix = right_matrix.view(batch_size, rank, c_in) # [batch_size, rank, c_in]

    # Imagine, that the output of `self.affine` (in SynthesisLayer) is N(0, 1)
    # Then, std of weights is sqrt(rank). Converting it back to N(0, 1)
    modulation = left_matrix @ right_matrix / np.sqrt(rank) # [batch_size, c_out, c_in]

    if not noise is None:
        std = torch.std(modulation,dim=[0,1],keepdim=True)
        noise = torch.randn(modulation.shape,device=x.device)*std
        modulation = modulation.add_(noise)

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5
    
    modulation *= scale

    W = weight.unsqueeze(0) * (modulation + 1.0) # [batch_size, c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=2, keepdim=True) + 1e-8) # [batch_size, c_out, c_in]
    W = W.to(dtype=x.dtype)

    # out = torch.einsum('boi,bihw->bohw', W, x)
    # x = x.view(batch_size, c_in, N) # [batch_size, c_in, h * w]
    x = x.permute(0,2,1) # [batch_size, c_in, N]
    out = torch.bmm(W, x) # [batch_size, c_out, N]
    # out = out.view(batch_size, c_out, h, w) # [batch_size, c_out, h, w]
    
    out = out.permute(0,2,1) # [batch_size, N, c_in]

    return out

class SpatialMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()



        self.network = nn.Sequential(nn.Linear(z_dim+36, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z, x_pe):
        z = torch.cat([z.unsqueeze(1).repeat(1,x_pe.shape[1],1),x_pe],dim=-1)
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()



        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


class HyperMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim_freq,map_output_dim_phase,scale=0.25):
        super().__init__()

        self.map_output_dim_freq = map_output_dim_freq
        self.map_output_dim_phase = map_output_dim_phase
        self.scale = scale

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim_freq+map_output_dim_phase))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        # frequencies_offsets *= self.scale
        frequencies = frequencies_offsets[..., :self.map_output_dim_freq]
        phase_shifts = frequencies_offsets[..., self.map_output_dim_freq:]

        return frequencies, phase_shifts


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

# network architecture from pi-gan
class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation='sin'):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.activation = activation

    def forward(self, x, freq, phase_shift=None,random_phase=False):
        x = self.layer(x)
        if not freq.shape == x.shape:
            freq = freq.unsqueeze(1).expand_as(x)
        if not phase_shift is None and not phase_shift.shape == x.shape:
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        if phase_shift is None:
            phase_shift = 0
        if self.activation == 'sin':
            if random_phase:
                phase_shift = phase_shift*torch.randn(x.shape[0],x.shape[1],1).to(x.device)
            return torch.sin(freq * x + phase_shift)
        else:
            return F.leaky_relu(freq * x + phase_shift, negative_slope=0.2)

class FMMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        # self.sin = Sine()

    def forward(self, x, freq, phase_shift,activation='sin',module_noise=None,weight_noise=None,gamma_noise=None):
        if weight_noise is not None:
            x = fmm_modulate_linear(x, self.layer.weight+weight_noise, freq, noise=module_noise, activation="sigmoid")
        else:
            x = fmm_modulate_linear(x, self.layer.weight, freq, noise=module_noise, activation="sigmoid")

        freq_shift = phase_shift[...,:self.output_dim]
        if gamma_noise is not None:
            std = torch.std(freq_shift,dim=-1,keepdim=True)
            noise = torch.randn(freq_shift.shape,device=x.device)*std
            freq_shift += noise
        if activation == 'sin':
            phase_shift_ = (phase_shift[...,self.output_dim:]-30)/15
        else:
            phase_shift_ = phase_shift[...,self.output_dim:] - 1

        freq_shift = freq_shift.unsqueeze(1)
        phase_shift_ = phase_shift_.unsqueeze(1)

        if activation == 'sin':
            return torch.sin(freq_shift*x+phase_shift_)
        elif activation == 'lrelu':
            return self.lrelu(freq_shift*x+phase_shift_)
        else:
            return x


class TALLSIREN(nn.Module):
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30

        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = self.color_layer_linear(rbg)

        return torch.cat([rbg, sigma], dim=-1)


class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor
    
    def forward_var(self, variances):
        return variances * self.scale_factor**2
    
    def forward_inv(self,coordinates):
        return coordinates / self.scale_factor



# SIREN network 'model': 'SPATIALSIRENMULTI_NEW',
class SPATIALSIRENMULTI_NEW(nn.Module):
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None,**kwargs):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])

        self.output_sigma = nn.ModuleList([
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
        ])
        
        self.color_layer_sine = nn.ModuleList([FiLMLayer(hidden_dim + 3, hidden_dim)])

        self.output_color = nn.ModuleList([
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
        ])
        
        # z_id and z_noise, 80 + 80
        self.mapping_network = CustomMappingNetwork(80 * 2, 256, (len(self.network) + len(self.color_layer_sine))*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.output_sigma.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.output_color.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, freq=None, phase=None, **kwargs):
        if freq is None:
            frequencies, phase_shifts = self.mapping_network(z)
            return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
        else:
            return self.forward_with_frequencies_phase_shifts(input, frequencies=freq, phase_shifts=phase, ray_directions=ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
        sigma = 0
        rgb = 0
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
            if index > 0:
                layer_sigma = self.output_sigma[index-1](x)
                if not index == 7:
                    layer_rgb_feature = x 
                else:
                    layer_rgb_feature = self.color_layer_sine[0](torch.cat([ray_directions, x], dim=-1),\
                        frequencies[..., len(self.network)*self.hidden_dim:(len(self.network)+1)*self.hidden_dim], phase_shifts[..., len(self.network)*self.hidden_dim:(len(self.network)+1)*self.hidden_dim])
                layer_rgb = self.output_color[index-1](layer_rgb_feature)

                sigma += layer_sigma
                rgb += layer_rgb


        rgb = torch.sigmoid(rgb)
        
        return torch.cat([rgb, sigma], dim=-1)


def geometry_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_output = m.weight.size(0)
            m.weight.normal_(0,np.sqrt(2/num_output))
            nn.init.constant_(m.bias,0)

def geometry_init_last_layer(radius):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                nn.init.constant_(m.weight,10*np.sqrt(np.pi/num_input))
                nn.init.constant_(m.bias,-radius)
    return init


# compute intersections 'model_sample': 'SPATIALSAMPLERELU', 
class SPATIALSAMPLERELU(nn.Module):
    def __init__(self, input_dim=3, z_dim=100, hidden_dim_sample=64, output_dim=1, device=None,**kwargs):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim_sample
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim_sample),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_sample, hidden_dim_sample),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_sample, hidden_dim_sample),
            nn.ReLU(inplace=True),
            # nn.Linear(32, 32),
            # nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Linear(hidden_dim_sample, 1)

        self.network.apply(geometry_init)
        self.output_layer.apply(geometry_init_last_layer(kwargs['init_radius']))
        if not 'center' in kwargs:
            self.center = torch.tensor([0,0,-1.5])
        else:
            self.center = torch.tensor(kwargs['center'])
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def calculate_intersection(self, intervals, vals, levels):
        intersections = []
        is_valid = []
        for interval,val,l in zip(intervals,vals,levels):
            x_l = interval[:,:,0]
            x_h = interval[:,:,1]
            s_l = val[:,:,0]
            s_h = val[:,:,1]
            # intersect = ((s_h-l)*x_l + (l-s_l)*x_h)/(s_h-s_l) #[batch,N_rays,3]
            # with torch.no_grad():
            #     scale = (s_h-s_l)
            scale = torch.where(torch.abs(s_h-s_l) > 0.05, s_h-s_l, torch.ones_like(s_h)*0.05)
            intersect = torch.where(((s_h-l<=0)*(l-s_l<=0)) & (torch.abs(s_h-s_l) > 0.05),((s_h-l)*x_l + (l-s_l)*x_h)/scale,x_h)
            # if distance is too small, choose the right point
            intersections.append(intersect)
            is_valid.append(((s_h-l<=0)*(l-s_l<=0)).to(intersect.dtype))
        
        return torch.stack(intersections,dim=-2),torch.stack(is_valid,dim=-2) #[batch,N_rays,level,3]

    def calculate_intersection_with_deform_(self, intervals, vals, deforms, levels):
        intersections_deform = []
        intersections_canonical = []
        is_valid = []
        for interval,val, d_vec, l in zip(intervals,vals,deforms,levels):
            x_l = interval[:,:,0]
            x_h = interval[:,:,1]
            s_l = val[:,:,0]
            s_h = val[:,:,1]
            d_l = d_vec[:,:,0]
            d_h = d_vec[:,:,1]
            # scale = torch.where(torch.abs(s_h-s_l) > 0.05, s_h-s_l, torch.ones_like(s_h)*0.05)
            # mask = ((s_h-l<=0)*(l-s_l<=0)) & (torch.abs(s_h-s_l) > 0.05)
            # intersect_deform = torch.where(mask == True, ((s_h-l)*x_l + (l-s_l)*x_h)/scale,x_h)
            # avg_deform = torch.where(mask == True, ((s_h-l)*d_l + (l-s_l)*d_h)/scale, d_h)
            # avg_density = torch.where(mask == True, ((s_h-l)*c_d_l + (l-s_l)*c_d_h)/scale, c_d_h)
            # avg_rgb = torch.where(mask == True, ((s_h-l)*c_rgb_l + (l-s_l)*c_rgb_h)/scale, c_rgb_h)


            scale = s_h-s_l#torch.where(torch.abs(s_h-s_l) > 0.05, , torch.ones_like(s_h)*0.05)
            mask = ((s_h-l<=0)*(l-s_l<=0)) #& (torch.abs(s_h-s_l) > 0.05)
            intersect_deform = torch.where(mask == True, ((s_h-l)*x_l + (l-s_l)*x_h)/scale,x_h)
            avg_deform = torch.where(mask == True, ((s_h-l)*d_l + (l-s_l)*d_h)/scale, d_h)

            # intersect_deform = torch.where(((s_h-l<=0)*(l-s_l<=0)) & (torch.abs(s_h-s_l) > 0.05),((s_h-l)*x_l + (l-s_l)*x_h)/scale,x_h)
            # avg_deform = torch.where(((s_h-l<=0)*(l-s_l<=0)) & (torch.abs(s_h-s_l) > 0.05),((s_h-l)*d_l + (l-s_l)*d_h)/scale, d_h)
            # avg_density = torch.where(((s_h-l<=0)*(l-s_l<=0)) & (torch.abs(s_h-s_l) > 0.05),((s_h-l)*c_d_l + (l-s_l)*c_d_h)/scale, c_d_h)
            # avg_rgb = torch.where(((s_h-l<=0)*(l-s_l<=0)) & (torch.abs(s_h-s_l) > 0.05),((s_h-l)*c_rgb_l + (l-s_l)*c_rgb_h)/scale, c_rgb_h)


            intersect_canonical = intersect_deform + avg_deform


            # if distance is too small, choose the right point
            intersections_deform.append(intersect_deform)
            intersections_canonical.append(intersect_canonical)
            is_valid.append(((s_h-l<=0)*(l-s_l<=0)).to(intersect_deform.dtype))
        
        return torch.stack(intersections_deform,dim=-2), torch.stack(intersections_canonical, dim=-2), torch.stack(is_valid,dim=-2)
        #[batch,N_rays,level,3]
    
    def forward(self,input):
        

        x = input
        # x += torch.tensor([[0,0,1.5]]).to(x.device)
        x = self.network(x)
        s = self.output_layer(x)

        return s

    def get_intersections_with_deform_with_(self, wp_sample_deform, wp_sample_canonic, vec_deform2canonic, levels, **kwargs):
        '''
        The func for 
        '''
        # levels num_l
        batch,N_rays,N_points,_ = wp_sample_canonic.shape
        
        w_points_canonic = wp_sample_canonic.reshape(batch,-1,3)
        w_points_canonic = self.gridwarper(w_points_canonic)

        # This is because the iossurfaces of SIREN is two sides. move them to use half of the isosurfaces
        w_points_canonic = w_points_canonic - self.center.to(w_points_canonic.device)

        # use a light wight MLP network to process a point x, and predict a scalar value s
        w_points_canonic = self.network(w_points_canonic)
        s = self.output_layer(w_points_canonic)
        # we can obtain the sigma value before the deformation network now

        s = s.reshape(batch,N_rays,N_points,1)
        s_l = s[:,:,:-1] 
        s_h = s[:,:,1:]

        # cost 
        cost = torch.linspace(N_points-1, 0, N_points-1).float().to(w_points_canonic.device).reshape(1,1,-1,1)
        # shape is batch_size x 1 x [N_points - 1] x 1, ranges from N_points -1, 0

        x_interval = []
        s_interval = []
        deform_interval = []

        for l in levels:
            r = torch.sign((s_h-l)*(l-s_l)) # [batch,N_rays,N_points-1]
            # on the two sides of the plane, the sign is negative. when the two sides across the plane, the sign is positive.
            r = r*cost
            _, indices = torch.max(r,dim=-2,keepdim=True)
            indices_expand = indices.expand(-1, -1, -1, 3)
            x_l_select = torch.gather(wp_sample_deform, -2, indices_expand) # [batch,N_rays,1]
            x_h_select = torch.gather(wp_sample_deform, -2, indices_expand+1) # [batch,N_rays,1]
            deform_l_select = torch.gather(vec_deform2canonic, -2, indices_expand) # [batch,N_rays,1]
            deform_h_select = torch.gather(vec_deform2canonic, -2, indices_expand+1) # [batch,N_rays,1]
            s_l_select = torch.gather(s_l,-2,indices)
            s_h_select = torch.gather(s_h,-2,indices)
            
            # gather the x coordinates and scalar
            x_interval.append(torch.cat([x_l_select,x_h_select],dim=-2))
            s_interval.append(torch.cat([s_l_select,s_h_select],dim=-2))
            deform_interval.append(torch.cat([deform_l_select, deform_h_select], dim=-2))

        # The intersections between deform space and rays
        
        intersections_deform, intersections_canonical, is_valid = self.calculate_intersection_with_deform_(x_interval, s_interval, deform_interval, \
                                                                                                        levels)
 
        # intersections = self.gridwarper.forward_inv(intersections)
        
        return intersections_deform, intersections_canonical, s, is_valid

    def get_intersections(self, input, levels, **kwargs):
        # levels num_l
        batch,N_rays,N_points,_ = input.shape
        
        x = input.reshape(batch,-1,3)
        x = self.gridwarper(x)

        # x += torch.tensor([[0,0,1.5]]).to(x.device)
        # x[...,2] += 1
        # x = x + torch.tensor([[0,0,1.5]]).to(x.device)
        x = x - self.center.to(x.device)

        # use a light wight MLP network to process a point x, and predict a scalar value s
        x = self.network(x)
        s = self.output_layer(x)

        s = s.reshape(batch,N_rays,N_points,1)
        s_l = s[:,:,:-1]
        s_h = s[:,:,1:]

        # cost 
        cost = torch.linspace(N_points-1,0,N_points-1).float().to(input.device).reshape(1,1,-1,1)
        # shape is batch_size x 1 x [N_points - 1] x 1, ranges from N_points -1, 0

        x_interval = []
        s_interval = []
        for l in levels:
            r = torch.sign((s_h-l)*(l-s_l)) # [batch,N_rays,N_points-1]
            # on the two sides of the plane, the sign is negative. when the two sides across the plane, the sign is positive.
            r = r*cost
            _, indices = torch.max(r,dim=-2,keepdim=True)
            x_l_select = torch.gather(input,-2,indices.expand(-1, -1, -1, 3)) # [batch,N_rays,1]
            x_h_select = torch.gather(input,-2,indices.expand(-1, -1, -1, 3)+1) # [batch,N_rays,1]
            s_l_select = torch.gather(s_l,-2,indices)
            s_h_select = torch.gather(s_h,-2,indices)
            # gather the x coordinates and scalar
            x_interval.append(torch.cat([x_l_select,x_h_select],dim=-2))
            s_interval.append(torch.cat([s_l_select,s_h_select],dim=-2))
        
        intersections,is_valid = self.calculate_intersection(x_interval,s_interval,levels)

        # intersections = self.gridwarper.forward_inv(intersections)
        
        return intersections,s,is_valid


class SPATIAL_SIREN_DEFORM(nn.Module):
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None, phase_noise=False, hidden_z_dim=128,  **kwargs):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.phase_noise = phase_noise

        self.network = nn.ModuleList([
            FiLMLayer(self.input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            # FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.epoch = 0
        self.step = 0
        
        self.network.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        self.final_layer = nn.Linear(hidden_dim, self.output_dim)
        self.final_layer.apply(frequency_init(25))

        if not kwargs.get('debug_control_seed', False):
            nn.init.zeros_(self.final_layer.weight)  # prevent large deformation at the start of training
        else:
            print('-------------- the final layer of the deformation network is not init as zero')


        self.mapping_network = CustomMappingNetwork(self.z_dim, hidden_z_dim, len(self.network)*hidden_dim*2)

        self.gridwarper = UniformBoxWarp(0.24) 

        self.deformation_scale = 1.0
    



    def forward(self, z_id, z_exp, coords, **kwargs):
        z = torch.cat([z_id, z_exp], dim=-1)
        batch_size, num_points, num_steps, _ = coords.shape
        batch_size = z_exp.shape[0]
        coords = coords.view(batch_size, -1, 3)

        coords = self.gridwarper(coords)

        # this is because we tried to use deform_ref as input, then find it doesn't help and cause the result flickering
        # thus remove here, should not affect performance
        add = torch.zeros((batch_size, num_points * num_steps, 4)).to(coords.device)
        input = torch.cat([coords, add], dim=-1)
        frequencies, phase_shifts = self.mapping_network(z)
        deformation = self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, **kwargs)
        new_coords = coords + deformation # deform into template space

        new_coords = new_coords.view(batch_size, num_points, num_steps, 3)
        deformation = deformation.view(batch_size, num_points, num_steps, 3)
        return new_coords, deformation

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, **kwargs):
        frequencies = frequencies*15 + 30

        x = input
        
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end],random_phase=self.phase_noise)
        
        x = self.final_layer(x)
        deformation = x[...,:3]
        return deformation