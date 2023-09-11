from asyncio import FIRST_COMPLETED
from logging import shutdown
import os
from matplotlib.pyplot import prism
import numpy as np
import math
from collections import deque

from yaml import parse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import torchvision.transforms as transforms
import importlib
import time 
import glob, shutil
from scipy.io import loadmat
# from generators import generators_neutex as generators
from siren import siren
import curriculums
import copy, plyfile
from torch_ema import ExponentialMovingAverage
import pytorch3d
from torch_ema import ExponentialMovingAverage
import argparse
import util as util
device = torch.device('cuda') #if torch.cuda.is_available() else 'cpu')
from PIL import Image 
import skvideo

def visualization(img_size, intersections_deform, is_valid, opt, step, intersections_canonic):
    # save intersections
    import matplotlib.pyplot as plt
    intersections_deform = intersections_deform[0].cpu().numpy()
    is_valid = is_valid[0].cpu().numpy()

    intersections_deform = intersections_deform.reshape(img_size,img_size,-1,3)
    is_valid = is_valid.reshape(img_size,img_size,-1,1)

    slice_yz = intersections_deform[:,img_size//2,:,1:].reshape(-1,2)
    slice_valid = is_valid[:,img_size//2,:,0].reshape(-1)

    plt.figure()
    plt.scatter(slice_yz[slice_valid.astype(np.bool),1],slice_yz[slice_valid.astype(np.bool),0],s=3,c='red')
    plt.axis('equal')
    plt.savefig(os.path.join(opt.output_dir,'%06d_deform_slice_intersection_yz.png'%step))
    plt.close()

    slice_xz = intersections_deform[img_size//2,:,:,::2].reshape(-1,2)
    slice_valid = is_valid[img_size//2,:,:,0].reshape(-1).astype(np.bool)
    plt.scatter(slice_xz[slice_valid,1],slice_xz[slice_valid,0],s=3,c='red')
    plt.axis('equal')
    # plt.scatter(slice_yz[:,1],slice_yz[:,0],s=3,c='red')
    plt.savefig(os.path.join(opt.output_dir,'%06d_deform_slice_intersection_xz.png'%step))
    plt.close()

    intersections_canonic = intersections_canonic[0].cpu().numpy()
    intersections_canonic = intersections_canonic.reshape(img_size,img_size,-1,3)

    slice_yz = intersections_canonic[:,img_size//2,:,1:].reshape(-1,2)
    slice_valid = is_valid[:,img_size//2,:,0].reshape(-1)

    plt.figure()
    plt.scatter(slice_yz[slice_valid.astype(np.bool),1],slice_yz[slice_valid.astype(np.bool),0],s=3,c='red')
    plt.axis('equal')
    plt.savefig(os.path.join(opt.output_dir,'%06d_canonic_slice_intersection_yz.png'%step))
    plt.close()

    slice_xz = intersections_canonic[img_size//2,:,:,::2].reshape(-1,2)
    slice_valid = is_valid[img_size//2,:,:,0].reshape(-1).astype(np.bool)
    plt.scatter(slice_xz[slice_valid,1],slice_xz[slice_valid,0],s=3,c='red')
    plt.axis('equal')
    # plt.scatter(slice_yz[:,1],slice_yz[:,0],s=3,c='red')
    plt.savefig(os.path.join(opt.output_dir,'%06d_canonic_slice_intersection_xz.png'%step))
    plt.close()

def save_ply(face_vertex, faces, filename):
    face_vertex = face_vertex.cpu().numpy()[0,...] * 0.1
    # rescale to 0.1
    print("face_vertex shape", face_vertex.shape)
    print("faces shape", faces.shape)
    
    num_verts = face_vertex.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(face_vertex[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)

def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()

def staged_forward(z_id, z_exp, noise, generator_ddp, deform_ddp, neutral_face_flag, stage, alpha, metadata, opt):
    '''
    real_imgs - 
    generator_ddp - 
    ema, ema2 - 
    alpha - the prograssive growing factor, either 1 or below 1
    scalar - the image scale ? 
    metadata - config files
    '''
    psi = 0.7
    device = z_exp.device
    img_size = metadata['img_size']
    batch_size = z_exp.shape[0]

    split_batch_size = z_exp.shape[0] // metadata['batch_split'] # minibatch split for memory reduction
    # batch split - the number of splited batches
    with torch.no_grad():
        pixels_all = []
        depth_all = []
        pose_all = []
        intersections_deform_all = []
        intersections_canonic_all = []
        is_valid_all = []
        weight_list = []
        for split in range(metadata['batch_split']):
            subset_z_exp = z_exp[split * split_batch_size:(split+1) * split_batch_size]
            subset_z_id = z_id[split * split_batch_size:(split+1) * split_batch_size]
            subset_noise = noise[split * split_batch_size:(split+1) * split_batch_size]
            # ------------------------------------------ obtain 3dmm neutral face here-------------------------------------------
            t = time.time()
            
            #print("generate face takes", time.time() - t)

            z = torch.cat([subset_z_id, subset_noise], dim=1)
            batch_size = subset_z_exp.size()[0]

            raw_frequencies, raw_phase_shifts = generator_ddp.siren.mapping_network(z)
            if not psi == 1:
                truncated_frequencies = generator_ddp.avg_frequencies + psi * (raw_frequencies - generator_ddp.avg_frequencies)
                truncated_phase_shifts = generator_ddp.avg_phase_shifts + psi * (raw_phase_shifts - generator_ddp.avg_phase_shifts)
                # print("psi not = 1")
            else:
                truncated_frequencies = raw_frequencies
                truncated_phase_shifts = raw_phase_shifts

            with torch.no_grad():
                wp_sample_deform, wp_inter_back_deform, levels, w_ray_origins, w_ray_directions, pitch, yaw, cam2world = generator_ddp.generate_points(subset_z_exp.size()[0], subset_z_exp.device, **metadata)
            t = time.time()
            bs, N_rays, N_steps, _ = wp_sample_deform.size()
            t2 = time.time()
            #print("generate data time = ", t2-t)
            gen_positions, output, intersections_deform, intersections_canonical, is_valid = \
                                                                    generator_ddp.forward(subset_z_id, subset_z_exp, subset_noise, \
                                                                        wp_sample_deform, wp_inter_back_deform, levels, w_ray_origins, w_ray_directions, pitch, yaw, \
                                                                            neutral_face_flag, deform_ddp, alpha, metadata, \
                                                                            freq=truncated_frequencies, phase=truncated_phase_shifts, stage_forward_flag=True)
            
            gen_imgs, depth, weights, transparency = output

            batch_size, N, num_steps, _ = intersections_deform.size()
            weighted_points = torch.sum(weights * intersections_deform, -2).view(batch_size, -1, 3)

            bs = gen_imgs.size()[0]
            weighted_points = weighted_points.view(bs, metadata['img_size'], metadata['img_size'], 3)

            pixels_all.append(gen_imgs)
            weight_list.append(weights)

            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous()
            depth_all.append(depth_map)
            gen_positions = torch.cat([pitch, yaw], -1)
            pose_all.append(gen_positions)

            intersections_deform_all.append(intersections_deform)
            intersections_canonic_all.append(intersections_canonical)
            is_valid_all.append(is_valid)

        weight_list_cat = torch.cat([p for p in weight_list], dim=0)
        pixels_all_cat = torch.cat([p for p in pixels_all], dim=0) # 16 x 64 x 64 x 3
        pixels_all_cat = pixels_all_cat.cpu()
        depth_all_cat = torch.cat([p for p in depth_all], dim=0)
        depth_all_cat = depth_all_cat.cpu()

        pose_all_cat = torch.cat([p for p in pose_all], dim=0)

        intersections_deform_cat = torch.cat([p for p in intersections_deform_all], dim=0)
        intersections_canonic_cat = torch.cat([p for p in intersections_canonic_all], dim=0)
        is_valid_all_cat = torch.cat([p for p in is_valid_all], dim=0)

        return pixels_all_cat, depth_all_cat, intersections_deform_cat, intersections_canonic_cat, is_valid_all_cat, weight_list_cat, weighted_points

def generate_img(generator, deform_net, z_id, z_exp, z_noise, flag, kwargs):
    with torch.no_grad():
        img, depth_map, intersections_deform, intersections_canonic, _, weight, weighted_points = staged_forward(z_id, z_exp, z_noise, generator, deform_net, flag, stage=kwargs['img_size'], alpha=1, metadata=kwargs, opt=opt)
        tensor_img = img.detach()
        
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()

    return img, tensor_img

transform = transforms.Compose(
[transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
        # torch.randn - sample random numbers from a normal distribution with mean 0 and varaiance 1
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
        # torch.rand - sample random numbers froma uniform distribution 
    return z

def sample_latent_id(bs, device, vae_net_id, metadata):
    with torch.no_grad():# 80-identity, 64-expression
        normal_id = z_sampler((bs, metadata['latent_dim']), device=device, dist='gaussian')
        z_id = vae_net_id.decode(normal_id)
    return z_id

def sample_latent_exp(bs, device, vae_net_exp, metadata):
    with torch.no_grad():# 80-identity, 64-expression
        normal_exp = z_sampler((bs, metadata['latent_dim']), device=device, dist='gaussian')
        z_exp = vae_net_exp.decode(normal_exp)
    return z_exp

def sample_latents(bs, device, vae_net_id, vae_net_exp, metadata):
    with torch.no_grad():# 80-identity, 64-expression
        normal_id = z_sampler((bs, metadata['latent_dim']), device=device, dist='gaussian')
        normal_exp = z_sampler((bs, metadata['latent_dim']), device=device, dist='gaussian')
        z_id = vae_net_id.decode(normal_id)
        z_exp = vae_net_exp.decode(normal_exp)
    return z_id, z_exp

def transform_exp(vae_net_exp, latent_exp):
    with torch.no_grad():# 80-identity, 64-expression
        z_exp = vae_net_exp.forward(latent_exp)[0]
    return z_exp
    
def read_latents(name):
    # load the latent codes for id, expression and so on.
    
    '''
        the data structure of ffhq_pose
        id : the identity code 1 x 80
        exp : the expression code 1 x 64
        tex : the texture code 1 x 80
        angle: 1 x 3, rotation x y z
        gamma: lighting code 1 x 27
        trans: 1 x 3, translation x y z
        lm68: the 68 keypoints 
    '''
    latents = loadmat(name)
    latent_id = torch.from_numpy(latents['id']).float()[0,...]
    latent_exp = torch.from_numpy(latents['exp']).float()[0,...]
    
    return latent_id, latent_exp

def transform_id(vae_net_id, latent_id):
    with torch.no_grad():# 80-identity, 64-expression
        z_id = vae_net_id.forward(latent_id)[0]
    return z_id


def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


if __name__ == '__main__':

    inter_i  = 22

    parser = argparse.ArgumentParser()
    # parser.add_argument('path', type=str)
    parser.add_argument('--generator_file', type=str, default='results/20220301-064028_warm_up_deform_0_switch_interval_2_DIF_lambda_4000_ths_0.000100_gen_gt/generator.pth')
    parser.add_argument('--deform_file', type=str, default='results/20220301-064028_warm_up_deform_0_switch_interval_2_DIF_lambda_4000_ths_0.000100_gen_gt/dif.pth')
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--output_dir', type=str, default='multiview_imgs/20220301-064028_sampleexp')
    parser.add_argument('--max_batch_size', type=int, default=1200000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--curriculum', type=str, default='SPATIALSIRENBASELINEGRAM_deform')
    parser.add_argument('--gen_points_threshold', type=float, default=0.00005, help='the gen points threshold')
    parser.add_argument('--sample_3dmm', type=float, default=1.0, help='the gen points threshold')
    parser.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    #facerecon
    parser.add_argument('--checkpoints_dir', type=str, default='./FaceRecon_Pytorch/checkpoints', help='models are saved here')
    parser.add_argument('--vis_batch_nums', type=float, default=1, help='batch nums of images for visulization')
    parser.add_argument('--eval_batch_nums', type=float, default=float('inf'), help='batch nums of images for evaluation')
    parser.add_argument('--use_ddp', type=util.str2bool, nargs='?', const=True, default=True, help='whether use distributed data parallel')
    parser.add_argument('--ddp_port', type=str, default='12355', help='ddp port')
    parser.add_argument('--display_per_batch', type=util.str2bool, nargs='?', const=True, default=True, help='whether use batch to show losses')
    parser.add_argument('--add_image', type=util.str2bool, nargs='?', const=True, default=True, help='whether add image to tensorboard')
    parser.add_argument('--world_size', type=int, default=1, help='batch nums of images for evaluation')

    # model parameters
    parser.add_argument('--model', type=str, default='facerecon', help='chooses which model to use.')

    # additional parameters
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

    # self.initialized = True
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
    parser.add_argument('--init_path', type=str, default='./FaceRecon_Pytorch/checkpoints/init_model/resnet50-0676ba61.pth')
    parser.add_argument('--use_last_fc', type=util.str2bool, nargs='?', const=True, default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./FaceRecon_Pytorch/BFM')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)
    parser.add_argument('--to_gram', type=str, default='v1')
    parser.add_argument('--gen_video', action='store_true', help='whether generate video')
    parser.add_argument('--use_depth', action='store_true', help='whether use depth loss for geomotry generation')
    opt = parser.parse_args()
    
    os.makedirs(opt.output_dir, exist_ok=True)

    # generator = torch.load(opt.path, map_location=torch.device(device))

    # load configs
    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)
    
    generators = importlib.import_module('generators.'+metadata['generator_module'])
    generator_core = getattr(siren, metadata['model']) # network structure for radiance field generation
    # generator = generators.ImplicitGenerator3d [generation_MPI_learn_hd file]
    generator = getattr(generators, metadata['generator'])(generator_core, metadata['latent_dim'],**metadata).to(device)
    print(opt.generator_file)
    generator.load_state_dict(torch.load(opt.generator_file, map_location=device))
    print("loaded generator")
    # generator = torch.load(opt.generator_file, map_location=device)
    generator.set_device(device)
    generator.eval()

    ema_file = opt.generator_file.split('generator')[0] + 'ema.pth'
    print(ema_file)
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema.load_state_dict(torch.load(ema_file, map_location=device))
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()

    # load deformnet
    # dif_net = importlib.import_module('siren.siren')
    # dif_model = getattr(dif_net, 'DeformedImplicitField')()

    # dif_net = importlib.import_module('dif_net.dif_net')
    # dif_model = getattr(dif_net, 'DeformedImplicitField')()

    dif_net = importlib.import_module('siren.siren')
    dif_model = getattr(dif_net, 'SPATIAL_SIREN_DEFORM')(input_dim=7, z_dim=64+80, output_dim=7)
    
    dif_model.load_state_dict(torch.load(opt.deform_file, map_location=device))
    print("loaded dif model")
    # print(dif_model)
    dif_model.eval()
    dif_model = dif_model.to(device)
    dif_model = dif_model.to(device)#.set_device(device)
    
    vae_net_id = importlib.import_module('VAE_model')
    vae_net_id = getattr(vae_net_id, 'VAE_ID')(80, 256)
    vae_net_id.load_state_dict(torch.load("./pretrained_vaes/identity/vae.pth", map_location='cpu'))
    print("load vae id")
    vae_net_id = vae_net_id.to(device)
    vae_net_id.eval()
    
    vae_net_exp = importlib.import_module('VAE_model')
    vae_net_exp = getattr(vae_net_exp, 'VAE_EXP')(64, 256)
    vae_net_exp.load_state_dict(torch.load("./pretrained_vaes/expression/vae.pth", map_location='cpu'))
    print("load vae exp")
    vae_net_exp = vae_net_exp.to(device)
    vae_net_exp.eval()

    options_dict = {
        'levels_start': 23,
        'levels_end': 8,
        'num_steps':24, #24
        'num_divs': 5,
        'batch_split': 1,
        'with_deform': True,
        # 'center':(0,0,0),
        'img_size':opt.image_size,
        'hierarchical_sample':True,
        'psi':0.7,
        'sample_dist': 'gaussian',
        'ray_start':0.88,
        'ray_end':1.12,
        'v_stddev': 0,
        'h_stddev': 0,
        'h_mean': math.pi/2,
        'v_mean': math.pi/2,
        'fov': 12,
        'lock_view_dependence': True,
        'white_back':False,
        'last_back': False,
        'clamp_mode': 'softplus',
        'rgb_clamp_mode': 'widen_sigmoid',
        'nerf_noise': 0,
        'max_batch_size' : opt.max_batch_size,
        'sigma_only' : False,
        'rgb_only' : -1,
        'use_pix_noise': False,
        'interval_scale': 0.1,
        'has_back': False,
        'delta_final': 1e10,
        'use_alpha': True
    }

    trajectory = []
    num_steps = 50
    yaw_all = np.linspace(-0.35,0.35, num_steps)
    for t in range(num_steps):
        # t=pose_ratio[19]
        pitch = math.pi/2 #0.2 * np.cos(t * 2 * math.pi) + math.pi/2
        yaw = yaw_all[t] + math.pi/2
        fov = 12
        trajectory.append((pitch, yaw, fov))

    seeds = [814, 1019, 1076]

    import torch.nn.functional as F
    kernel = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])/16
    GaussianBlurKernel = torch.zeros(3, 3, 3, 3)
    GaussianBlurKernel[0,0] = kernel
    GaussianBlurKernel[1,1] = kernel
    GaussianBlurKernel[2,2] = kernel


    mat_ids = [1, 6]
    exp_files = []
    identity_files = []
    
    latent_id, latent_exp = read_latents('./mat_files/01626.mat')
    ratio = np.linspace(0, 1.0, num_steps)
    for pp in range(num_steps):
        exp_files.append((ratio[pp]*latent_exp).unsqueeze(0))

    num_expression = len(exp_files)
    latent_exp = latent_exp.unsqueeze(0).to(device)
    import skvideo
    skvideo.setFFmpegPath("/usr/bin/")
    import skvideo.io
    from skvideo.io import FFmpegWriter
    with torch.no_grad():
        generator.generate_avg_frequencies(vae_net_id, vae_net_exp)
        for seed in seeds:
            flag = True
            images = []
            depths = []
            torch.manual_seed(seed)
            z_id, _ = sample_latents(1, device, vae_net_id, vae_net_exp, metadata)
            noise = z_sampler((1, 80), device=device, dist='gaussian') 

            cnt_output_dir = os.path.join(opt.output_dir, '%04d/'%seed)
            os.makedirs(cnt_output_dir, exist_ok=True)
            

            output_name = f'img_{seed}_.mp4'
            writer = FFmpegWriter(os.path.join(opt.output_dir, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'}, verbosity=10)
            frames = []
            for exp_id in range(num_steps):
                z_exp = exp_files[exp_id].to(device)
                pitch, yaw, fov = trajectory[exp_id]
                options_dict['v_mean'] = pitch 
                options_dict['h_mean'] = yaw   
                options_dict['h_stddev'] = 0
                options_dict['v_stddev'] = 0

                i = 0
                img, tensor_img = generate_img(generator, dif_model, z_id, z_exp, noise, False, options_dict)
                
                import PIL.ImageDraw as ImageDraw
                img = Image.new('L', (opt.image_size, opt.image_size), 0)
                bs, _, img_size, _ = tensor_img.size()
                save_image(tensor_img, os.path.join(cnt_output_dir, "pred_img_%04d_exp_%04d_%03d_.png"%(seed, exp_id, i)), normalize=True,range=(-1,1))
                i += 1
                frames.append(tensor_to_PIL(tensor_img))
            for frame in frames:
                writer.writeFrame(np.array(frame))
                
            writer.close()          