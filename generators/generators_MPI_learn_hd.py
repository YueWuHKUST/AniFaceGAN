"""Implicit generator for 3D volumes"""
#from dbm import _KeyType
import random
import torch.nn as nn
import torch
import torch.nn.functional as F

from .volumetric_rendering_MPI_learn import *
from siren import siren as siren_

class ImplicitGenerator3d(nn.Module):
    def __init__(self, siren, z_dim, **kwargs):
        super().__init__()
        '''
        siren - siren network architecture
        z - dim - the dimension of latent z
        kwargs - the configs
        '''
        self.z_dim = z_dim
        self.siren = siren(output_dim=4, z_dim=self.z_dim, input_dim=3, device=None,**kwargs)
        self.model_sample = getattr(siren_, kwargs['model_sample'])(device=None,**kwargs)
        self.epoch = 0
        self.step = 0
        kernel = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])/16
        self.GaussianBlurKernel = torch.zeros(3, 3, 3, 3)
        self.GaussianBlurKernel[0,0] = kernel
        self.GaussianBlurKernel[1,1] = kernel
        self.GaussianBlurKernel[2,2] = kernel
        self.first_deform=None

    def set_device(self, device):
        self.device = device
        self.siren = self.siren.to(device)
        self.model_sample = self.model_sample.to(device)

    def generate_points(self, bs, device, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, \
                    pitch=None,yaw=None, region=None, sample_dist=None, **kwargs):
        # scripts to sample points along a ray
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(bs, 64, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points_sample, _, transformed_ray_directions, transformed_ray_origins, pitch, yaw, cam2world = transform_sampled_points(points_cam, z_vals, rays_d_cam,pitch=pitch,yaw=yaw, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist, randomize=False)
            transformed_points_sample = transformed_points_sample.reshape(bs, img_size*img_size, -1, 3)
            levels = torch.linspace(kwargs['levels_start'], kwargs['levels_end'], num_steps -1).to(device)

            # intersection with the predefined background plane
            # assume the canoical space and the expression space share the same MPI back level
            transformed_points_back, _ = get_intersection_with_MPI(transformed_ray_directions,transformed_ray_origins,device=self.device,mpi_start=-0.12,mpi_end=-0.12,mpi_num=1)
        return transformed_points_sample, transformed_points_back, levels, transformed_ray_origins, transformed_ray_directions, pitch, yaw, cam2world

    def forward(self, subset_z_id, subset_z_exp, subset_noise, \
        wp_sample_deform, wp_inter_back_deform, levels, w_ray_origins, w_ray_directions, pitch, yaw, \
                        neutral_face_flag, deform_ddp, alpha, metadata, \
                                    freq=None, phase=None, stage_forward_flag=False):
        """
        Args:
            subset_z_id: the identity latent code
            subset_z_exp: the expression latent code
            wp_sample_deform: the sampled points in target space in world coorinate system
            wp_inter_back_deofrom: the intersections between rays and background plane in world coordinate system
            w_rays_orgins: the world rays origins
            w_ray_directions: the world rays directions
            pitch, yaw: camera pose
        """
        
        gen_positions = torch.cat([pitch, yaw], -1)
        bs = subset_z_id.size()[0]
        # Deform the points based on identity and expression, wp_sample_canonic is used for generate the levels, and deform vector is used for deform points from target to canonic
        wp_sample_canonic, w_vec_deform2canonic = deform_ddp(subset_z_id, subset_z_exp, wp_sample_deform)
        
        # Compute the intersections in target space and canonical space
        intersections_deform, intersections_canonical, _, is_valid = self.model_sample.get_intersections_with_deform_with_(wp_sample_deform, wp_sample_canonic, w_vec_deform2canonic, levels) # [batch,H*W,num_steps,3]
        transformed_points_canonical = torch.cat([intersections_canonical, wp_inter_back_deform],dim=-2)
        transformed_points_deform = torch.cat([intersections_deform, wp_inter_back_deform], dim=-2)
        is_valid = torch.cat([is_valid,torch.ones(is_valid.shape[0],is_valid.shape[1],1,is_valid.shape[-1]).to(is_valid.device)],dim=-2)
        # Use radiance generator to generate color and density
        output = self.generate(subset_z_id, subset_noise, transformed_points_deform, transformed_points_canonical, is_valid, \
            w_ray_origins, w_ray_directions, \
            stage=metadata['img_size'], alpha=alpha, freq=freq, phase=phase, stage_forward_flag=stage_forward_flag, **metadata)
        
        return gen_positions, output, transformed_points_deform, transformed_points_canonical, is_valid
        


    def generate(self, z_id, z_noise, transformed_points_deform, transformed_points_canonical, is_valid, \
        transformed_ray_origins, transformed_ray_directions,  \
            stage, alpha, freq, phase, stage_forward_flag, \
        img_size, num_steps, region=None, lock_view_dependence=False, **kwargs):
        batch_size = z_id.shape[0]
        if not 'delta_final' in kwargs:
            kwargs['delta_final'] = 1e10
        # print("lock_view_dependence", lock_view_dependence)
        # concat the points of implict MPIs, and the points on the back
        
        with torch.no_grad():
            z_vals_deform = torch.sqrt(torch.sum((transformed_points_deform - transformed_ray_origins.unsqueeze(2))**2,dim=-1,keepdim=True)) # [batch,H*W,num_steps,1]
            z_vals_deform[is_valid==0] = 10.
            
            # z_vals_canonical = torch.sqrt(torch.sum((transformed_points_canonical - transformed_ray_origins.unsqueeze(2))**2,dim=-1,keepdim=True)) # [batch,H*W,num_steps,1]
            # z_vals_canonical[is_valid==0] = 10.
            #z_vals is the distances, and denote invalid points's distance as 10

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, -1, 3)
            

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
                if lock_view_dependence:
                    transformed_ray_directions_expanded = transformed_ray_directions_expanded/torch.norm(transformed_ray_directions_expanded, dim=-1, keepdim=True)

        # intersections
        transformed_points_canonical_reshape = transformed_points_canonical.reshape(batch_size, -1, 3)
        
        # given intersections, and latent code z, and ray directions. Use siren to generate a coarse output
        # set z as the concatenation of z_id and z_noise
        z = torch.cat([z_id, z_noise], dim=1)
        if freq is None:
            coarse_output = self.siren(transformed_points_canonical_reshape, z, ray_directions=transformed_ray_directions_expanded, stage=stage, alpha=alpha).reshape(batch_size, -1, num_steps, 4)
        else:
            max_batch_size = 5000
            coarse_output = torch.zeros((batch_size, transformed_points_canonical_reshape.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points_canonical_reshape.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points_canonical_reshape[b:b+1, head:tail], freq[b:b+1], phase[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], stage=stage, alpha=alpha)
                    head += max_batch_size
            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)

        all_outputs = coarse_output
        all_z_vals_deform = z_vals_deform#z_vals_canonical#
        
        _, indices_deform = torch.sort(all_z_vals_deform, dim=-2)
        all_z_vals_deform = torch.gather(all_z_vals_deform, -2, indices_deform)
        all_outputs_deform = torch.gather(all_outputs, -2, indices_deform.expand(-1, -1, -1, 4))
        is_valid_deform = torch.gather(is_valid,-2,indices_deform)

        bg_pos_deform = torch.argmax(indices_deform,dim=-2)

        # the volume rendering process. Check how this is accomplished
        pixels_deform, depth_deform, weights_deform, T_deform = fancy_integration(all_outputs_deform, all_z_vals_deform, is_valid=is_valid_deform, bg_pos=bg_pos_deform, \
                                        use_alpha=kwargs.get('use_alpha', True), alpha_delta=kwargs.get('alpha_delta', 0.04), \
                                        device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), \
                                        clamp_mode=kwargs['clamp_mode'], rgb_clamp_mode=kwargs['rgb_clamp_mode'], noise_std=kwargs['nerf_noise'], \
                                        delta_final=kwargs['delta_final'])

        pixels_deform = pixels_deform.reshape((batch_size, img_size, img_size, 3))
        pixels_deform = pixels_deform.permute(0, 3, 1, 2).contiguous() * 2 - 1

        depth_deform = depth_deform.reshape((batch_size, img_size, img_size, 1))
        depth_deform = depth_deform.permute((0, 3, 1, 2))
        # print("pixels=", pixels_deform[0, :10, 0, 0])
        return pixels_deform, depth_deform, weights_deform, T_deform

    def generate_avg_frequencies(self, vae_net_id, vae_net_exp):
        with torch.no_grad():# 80-identity, 64-expression
            normal_id = torch.randn((10000, 256), device=self.device)
            normal_exp = torch.randn((10000, 256), device=self.device)
            z_id = vae_net_id.decode(normal_id)
            z_exp = vae_net_exp.decode(normal_exp)
            z_noise = torch.randn((10000, 80), device=self.device)
        z = torch.cat([z_id, z_noise], dim=1)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        self.avg_z_id = z_id.mean(0, keepdim=True)
        self.avg_z_exp = z_exp.mean(0, keepdim=True)
        self.avg_z_noise = z_noise.mean(0, keepdim=True)