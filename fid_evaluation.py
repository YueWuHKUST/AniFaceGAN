from __future__ import print_function
import os
import shutil
import torch

from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
from pytorch_fid import fid_score
import datasets
from tqdm import tqdm
import copy
import argparse
import curriculums

from generators import generators_MPI_learn_hd as generators
from siren import siren
import math
import time
import importlib

from logging import shutdown
import os
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
from siren import siren
import fid_evaluation
import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy
from torch_ema import ExponentialMovingAverage
import pytorch3d
# from loss import *
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
import argparse
import util
device = torch.device('cuda')
from PIL import Image 
import torch_fidelity

def staged_forward(fixed_exp_z, fixed_id_z, fixed_noise_z, generator_ddp, deform_ddp, vae_net_id, vae_net_exp, stage, alpha, metadata, opt):
    '''
    real_imgs - 
    generator_ddp - 
    ema, ema2 - 
    alpha - the prograssive growing factor, either 1 or below 1
    scalar - the image scale ? 
    metadata - config files
    '''
    device = fixed_exp_z.device
    img_size = metadata['img_size']
    batch_size = fixed_exp_z.shape[0]

    z_exp = fixed_exp_z
    z_id = fixed_id_z
    noise = fixed_noise_z
    neutral_face_flag = False 

    split_batch_size = z_exp.shape[0]  # minibatch split for memory reduction
    # batch split - the number of splited batches
    with torch.no_grad():
        pixels_all = []
        depth_all = []
        pose_all = []
        intersections_deform_all = []
        intersections_canonic_all = []
        is_valid_all = []

        for split in range(1):
            subset_z_exp = z_exp[split * split_batch_size:(split+1) * split_batch_size]
            subset_z_id = z_id[split * split_batch_size:(split+1) * split_batch_size]
            subset_noise = noise[split * split_batch_size:(split+1) * split_batch_size]
            # ------------------------------------------ obtain 3dmm neutral face here-------------------------------------------
            t = time.time()

            z = torch.cat([subset_z_id, subset_noise], dim=1)
            batch_size = subset_z_exp.size()[0]

            raw_frequencies, raw_phase_shifts = generator_ddp.siren.mapping_network(z)
            truncated_frequencies = raw_frequencies
            truncated_phase_shifts = raw_phase_shifts

            wp_sample_deform, wp_inter_back_deform, levels, w_ray_origins, w_ray_directions, pitch, yaw, _ = generator_ddp.generate_points(subset_z_exp.size()[0], subset_z_exp.device, **metadata)
            gen_positions, output, intersections_deform, intersections_canonical, is_valid = \
                                                                    generator_ddp.forward(subset_z_id, subset_z_exp, subset_noise, \
                                                                        wp_sample_deform, wp_inter_back_deform, levels, w_ray_origins, w_ray_directions, pitch, yaw, \
                                                                            neutral_face_flag, deform_ddp, alpha, metadata, \
                                                                            freq=truncated_frequencies, phase=truncated_phase_shifts, stage_forward_flag=True)
            gen_imgs, depth, weights, transparency = output
            pixels_all.append(gen_imgs)
        pixels_all_cat = torch.cat([p for p in pixels_all], dim=0) # 16 x 64 x 64 x 3
        pixels_all_cat = pixels_all_cat.cpu()

        return pixels_all_cat

def output_real_images(dataloader, num_imgs, real_dir):
    img_counter = 0
    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
    for i in range(num_imgs//batch_size):
        #print(next(dataloader))
        real_imgs, _ = next(dataloader)
        for img in real_imgs:
            save_image(img, os.path.join(real_dir, f'{img_counter:0>5}.png'), normalize=True, range=(-1, 1))
            img_counter += 1

def setup_evaluation(dataset_name, dataset, generated_dir, target_size=128):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join('EvalImages', dataset_name + '_real_images_' + str(target_size))
    dataset = getattr(datasets, metadata['dataset'])(opt, **metadata)
    dataloader, _ = datasets.get_dataset(dataset, batch_size=1)
    print(dataset)
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        print('outputting real images...')
        output_real_images(dataloader, 5000, real_dir)
        print('...done')

    os.makedirs(generated_dir, exist_ok=True)
    return real_dir#, dataloader

def output_images(dataloader, generator, deform, input_metadata, rank, world_size, output_dir, alpha, num_imgs=10):
    metadata = copy.deepcopy(input_metadata)
    metadata['img_size'] = 128
    metadata['batch_size'] = 4

    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 1

    img_counter = rank
    generator.eval()
    deform.eval()
    img_counter = rank

    if rank == 0: pbar = tqdm("generating images", total = num_imgs)

    batch_size = dataloader.batch_size
    dataloader = iter(dataloader)
        
    with torch.no_grad():
        while img_counter < num_imgs:
            _, _, _, _, _, _,  id_z, exp_z = next(dataloader)
            device = generator.module.device
            noise_z = torch.randn((metadata['batch_size'], 80), device=device)
            id_z = id_z.to(device)
            exp_z = exp_z.to(device)
            generated_imgs = staged_forward(exp_z, id_z, noise_z, generator, deform, 1.0, stage=input_metadata['img_size'], alpha=alpha, metadata=metadata)[0]
            for img in generated_imgs:
                save_image(img, os.path.join(output_dir, f'{img_counter:0>5}.png'), normalize=True, range=(-1, 1))
                img_counter += world_size
                if rank == 0: pbar.update(world_size)
    if rank == 0: pbar.close()

def calculate_fid(dataset_name, generated_dir, target_size=128):
    real_dir = os.path.join('EvalImages', 'real' + '_real_images_' + str(target_size))
    print(real_dir, generated_dir)
    for i in range(10):
        try:
            fid = fid_score.calculate_fid_given_paths([real_dir, generated_dir], 128, 'cuda', 2048)
            break
        except:
            print('failed to load evaluation images, try %02d times'%i)
            time.sleep(0.5)

    torch.cuda.empty_cache()

    return fid

def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
        # torch.randn - sample random numbers from a normal distribution with mean 0 and varaiance 1
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
        # torch.rand - sample random numbers froma uniform distribution 
    return z

def sample_latents(bs, device, vae_net_id, vae_net_exp, metadata):
    with torch.no_grad():# 80-identity, 64-expression
        normal_id = z_sampler((bs, metadata['latent_dim']), device=device, dist='gaussian')
        normal_exp = z_sampler((bs, metadata['latent_dim']), device=device, dist='gaussian')
        z_id = vae_net_id.decode(normal_id)
        z_exp = vae_net_exp.decode(normal_exp)
    return z_id, z_exp



if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_file', type=str, default='results/20220507-185825_warm_up_deform_2000_switch_interval_3_DIF_lambda_0_ths_0.000010/step115000_generator.pth')
    parser.add_argument('--deform_file', type=str, default='results/20220507-185825_warm_up_deform_2000_switch_interval_3_DIF_lambda_0_ths_0.000010/step115000_dif.pth')
#     parser.add_argument('discriminator_file', type=str)
    parser.add_argument('--output_dir', type=str, default='generate_imgs1k/20220510-123836_warm_up_deform_2000_switch_interval_3_DIF_lambda_0_ths_0.000010/')
    parser.add_argument('--curriculum', type=str, default='SPATIALSIRENBASELINEGRAM_deform_bs4split2_de1rgb1_t')
    parser.add_argument('--num_images', type=int, default=5000)
    parser.add_argument('--gpu_type', type=str, default='8000')
    parser.add_argument('--keep_percentage', type=float, default='1.0')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--max_batch_size', type=int, default=None)
    parser.add_argument('--debug_mode', action='store_true')

    parser.add_argument('--checkpoints_dir', type=str, default='./FaceRecon_Pytorch/checkpoints', help='models are saved here')
    parser.add_argument('--vis_batch_nums', type=float, default=1, help='batch nums of images for visulization')
    parser.add_argument('--eval_batch_nums', type=float, default=float('inf'), help='batch nums of images for evaluation')
    parser.add_argument('--use_ddp', type=util.str2bool, nargs='?', const=True, default=True, help='whether use distributed data parallel')
    parser.add_argument('--ddp_port', type=str, default='12355', help='ddp port')
    parser.add_argument('--display_per_batch', type=util.str2bool, nargs='?', const=True, default=True, help='whether use batch to show losses')
    parser.add_argument('--add_image', type=util.str2bool, nargs='?', const=True, default=True, help='whether add image to tensorboard')
    parser.add_argument('--world_size', type=int, default=1, help='batch nums of images for evaluation')

    parser.add_argument('--sample_3dmm', type=float, default=0.1, help='the gen points threshold')
    parser.add_argument('--gen_points_threshold', type=float, default=0.00005, help='the gen points threshold')

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

    if '6' in opt.gpu_type:
        max_batch_size = 2400000
    else:
        max_batch_size = 94800000

    if opt.max_batch_size != None:
        max_batch_size = opt.max_batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    curriculum = getattr(curriculums, opt.curriculum)
    curriculum['dataset'] = 'FFHQ128'
    metadata = curriculums.extract_metadata(curriculum, 0)
    metadata['img_size'] = 128
    metadata['batch_size'] = 4
    metadata['h_stddev'] = metadata.get('h_stddev_eval', metadata['h_stddev'])
    metadata['v_stddev'] = metadata.get('v_stddev_eval', metadata['v_stddev'])
    metadata['sample_dist'] = metadata.get('sample_dist_eval', metadata['sample_dist'])
    metadata['psi'] = 1.0
    metadata['num_steps'] = 24
    metadata['final_num_steps'] = 24
    metadata['nerf_noise'] = 0
    metadata['interval_scale'] = 1.
    metadata['has_back'] = True
    metadata['last_back'] = False
    metadata['white_back'] = False
    metadata['phase_noise'] = False
    metadata['delta_final'] = 1e10
    metadata['hierarchical_sample'] = 1
    metadata['lock_view_dependence'] = True
    metadata['train_coarse'] =  True
    metadata['levels_start'] = 23
    metadata['levels_end'] = 8
    metadata['use_alpha'] = True
    metadata['num_levels'] = metadata['num_steps'] - 1
    metadata['debug_mode'] = False

    real_images_dir = setup_evaluation("real", curriculum['dataset'], opt.output_dir, target_size=metadata['img_size'])

    os.makedirs(opt.output_dir, exist_ok=True)

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

    # from FaceRecon_Pytorch.models import create_model
    # bfm_model = create_model(opt, metadata)
    # bfm_model = bfm_model.to(device)
    # bfm_model.set_device(device)
    # bfm_model.eval()

    # assert(opt.num_images % discriminator_batch_size == 0)
    i = 0
    for img_counter in tqdm(range(opt.num_images)):
        torch.manual_seed(img_counter)
        
        with torch.no_grad():
            z_id, z_exp = sample_latents(1, device, vae_net_id, vae_net_exp, metadata)
            z_noise = z_sampler((1, 80), device=device, dist='gaussian') 
            img = staged_forward(z_exp, z_id, z_noise, generator, dif_model, vae_net_id, vae_net_exp, stage=128, alpha=1, metadata=metadata, opt=opt)[0]
            save_image(img, os.path.join(opt.output_dir, f'{img_counter:0>5}.png'), normalize=True, range=(-1, 1))
            # img = generator.staged_forward(z, max_batch_size=max_batch_size, randomize=True,stage=128,alpha=1, **metadata)[0].to(device)
        i += 1
    
    metrics_dict = torch_fidelity.calculate_metrics(input1=opt.output_dir, input2=real_images_dir, cuda=True, isc=True, fid=True, kid=True, ppl=False, verbose=True)
    print(metrics_dict)
