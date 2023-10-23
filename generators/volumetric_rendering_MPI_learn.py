import time
from functools import partial

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from .math_utils_torch import *

def perturb_points_near(points, ray_start, ray_end, num_steps, metadata):
    # start=0.88, end=1.12, split into num_steps
    single_step = (ray_end - ray_start) / num_steps
    device = points.device
    radius = single_step / metadata['num_divs']# num_division=5
    offset = (torch.rand(points.shape, device=device)-0.5) * radius
    points_near = points + offset
    return points_near

def fancy_integration(rgb_sigma, z_vals, device, is_valid, bg_pos, use_alpha=True, alpha_delta=0.04, noise_std=0.5, last_back=False, white_back=False, clamp_mode=None, rgb_clamp_mode=None, fill_mode=None, eps=1e-3, sigma_only=False, rgb_only=-1, delta_final=1e10):

    '''
    the volumn rendering process
    '''
    # color and density
    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]
    
    if not use_alpha:
        deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    else:
        # here, deltas is the distance beween sequential points
        deltas = torch.ones_like(z_vals[:, :, 1:] - z_vals[:, :, :-1])*alpha_delta
    delta_inf = delta_final * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2) # [batch,N_rays,num_steps,1]

   
    # change the deltas of background locations as inifinite
    bg_pos = F.one_hot(bg_pos.squeeze(-1),num_classes=deltas.shape[-2]).to(torch.bool) # [batch,N_rays,num_steps]
    bg_pos = bg_pos.unsqueeze(-1) # [batch,N_rays,num_steps,1]
    deltas[bg_pos] = delta_final

    noise = torch.randn(sigmas.shape, device=device) * 0#noise_std

    if not rgb_only==-1:
        sigmas[z_vals<rgb_only] = -1e5
        sigmas[z_vals>=rgb_only] = 1e5

    # sigma larger, alpha larger.
    if clamp_mode == 'softplus':
        alphas = 1-torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"
    
    if rgb_clamp_mode == 'sigmoid':
        pass
    elif rgb_clamp_mode == 'widen_sigmoid':
        rgbs = rgbs*(1+2*eps) - eps
    else:
        raise "Need to choose rgb clamp mode"
    
    if sigma_only:
        rgbs = torch.zeros_like(rgbs)
        rgbs[...,2] += 0.5
    
    alphas = alphas*is_valid

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    T = torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights = alphas * T
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    if white_back:
        bg_pos_ = bg_pos.repeat(1, 1, 1, 3)
        rgbs[bg_pos_] = 1

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)/weights_sum

    # if white_back:
    #     rgb_final = rgb_final + 1-weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)
    
    # T is transparency here
    return rgb_final, depth_final, weights, T


def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, ray directions in camera space."""
    
    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2) # = -9.5

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))

    # ray_start = 0.88
    # ray_end = 1.12
    # 1 x num_steps x 1 -> W*H x num_steps x 1
    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    # rays_d_cam = [256 * 256] * 3
    # 65536 * num_steps * 3
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals
    
    # [H * W] x 64 x 3
    # print("points.shape", points.size())
    #print("n = ", n)
    points = torch.stack(n*[points])
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)

    return points, z_vals, rays_d_cam

def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals

def perturb_points_2(points, z_vals, ray_directions, device, spread):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = torch.randn(z_vals.shape, device=device) * spread
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2).contiguous()
    return points, z_vals

def get_intersection_with_MPI(transformed_ray_directions,transformed_ray_origins,device, mpi_start=0.12,mpi_end=-0.12,mpi_num=24):

    mpi_z_vals = torch.linspace(mpi_start, mpi_end, mpi_num, device=device)

    z_vals = mpi_z_vals.view(1,1,mpi_num) - transformed_ray_origins[...,-1:] #[batch,N,mpi_num]

    z_vals = z_vals/transformed_ray_directions[...,-1:] #[batch,N,mpi_num]

    z_vals = z_vals.unsqueeze(-1)

    points = transformed_ray_origins.unsqueeze(2) + transformed_ray_directions.unsqueeze(2)*z_vals

    return points, z_vals

def transform_sampled_points(points, z_vals, ray_directions, device, pitch=None,yaw=None, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5, mode='normal',randomize=True):
    n, num_rays, num_steps, channels = points.shape

    camera_origin, pitch, yaw = sample_camera_positions(n=points.shape[0], pitch=pitch,yaw=yaw, r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)


    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    #print(homogeneous_origins)
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw, cam2world_matrix


def sample_camera_positions(device, n=1, pitch=None,yaw=None, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """Samples n random locations along a sphere of radius r. Uses a gaussian distribution for pitch and yaw"""
    if pitch is not None and yaw is not None:
        phi = pitch
        theta = yaw
    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

        theta = torch.clamp(theta, horizontal_mean-1.3,horizontal_mean+1.3)
        phi = torch.clamp(phi, vertical_mean-1.3,vertical_mean+1.3)
    else:
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)# torch.cuda.FloatTensor(n, 3).fill_(0)#torch.zeros((n, 3))

    # theta horizonal angle
    # phi vertical angle
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta


def get_camera_positions_with_pose(device, phi, theta,  n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    
    theta = (torch.sigmoid(theta)-0.5) * 2 * horizontal_stddev + horizontal_mean
    phi = (torch.sigmoid(phi)-0.5) * 2 * vertical_stddev + vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)# torch.cuda.FloatTensor(n, 3).fill_(0)#torch.zeros((n, 3))

    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta

def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""


    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world