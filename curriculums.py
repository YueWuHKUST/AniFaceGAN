import math

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    last_epoch = 0
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step:
            last_epoch = curriculum_step
    return last_epoch

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    # step = get_current_step(curriculum, epoch)
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict

SPATIALSIRENBASELINEGRAM_deform = {

    0: {'batch_size': 4, 'num_steps': 24, 'img_size': 128, 'batch_split': 4, 'batch_split_dif': 4, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'FFHQ128_3dmm_face', 'density_lambda': 20, 'rgb_lambda': 20},
    int(100e3):  {'batch_size': 4, 'num_steps': 24, 'img_size': 128, 'batch_split': 4, 'batch_split_dif': 4, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'FFHQ128_3dmm_face', 'density_lambda': 15, 'rgb_lambda': 15},

    # 0: {'batch_size': 8, 'num_steps': 12, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    # # int(50): {'batch_size': 2, 'num_steps': 12, 'img_size': 256, 'batch_split': 2, 'gen_lr': 1e-6, 'disc_lr': 1e-5},
    # int(100e3): {},
    'sample_points': 10000,
    'points_split': 10000,
    'fov': 12, # Camera field of view
    'ray_start': 0.88, # Near clipping for camera rays.
    'ray_end': 1.12, # Far clipping for camera rays.
    'levels_start':23,
    'levels_end': 8,
    'init_radius': 0,
    'fade_steps': 10000,
    'h_stddev': 0.3, # Stddev of camera yaw in radians.
    'v_stddev': 0.155, # Stddev of camera pitch in radians.
    'h_mean': math.pi*0.5, #Mean of camera yaw in radians.
    'v_mean': math.pi*0.5, #Mean of camera pitch in radians.
    'sample_dist': 'gaussian', #Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    'topk_interval': 2000, #Interval over which to fade the top k ratio.
    'topk_v': 0.6, # Minimum fraction of a batch to keep during top k training.
    'betas': (0, 0.9), # Beta parameters for Adam.
    'unique_lr': True, # Whether to use reduced LRs for mapping network.
    'weight_decay': 0, # Weight decay parameter.
    'r1_lambda': 1, #R1 regularization parameter.
    'latent_dim': 256, #Latent dim for Siren network  in generator.
    'hidden_dim': 256, 
    'hidden_dim_sample': 128,
    'grad_clip': 0.3, #Grad clipping parameter.
    'model': 'SPATIALSIRENMULTI_NEW', # Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    'model_sample': 'SPATIALSAMPLERELU', 
    'generator': 'ImplicitGenerator3d', # Generator class. (ImplicitGenerator3d)
    'generator_module': 'generators_MPI_learn_hd',
    'discriminator': 'ProgressiveEncoderDiscriminatorAntiAlias', # Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    'clamp_mode': 'softplus', # Clamping function for Siren density output. (relu | softplus)
    'rgb_clamp_mode': 'widen_sigmoid', 
    'z_dist': 'gaussian', # Latent vector distributiion. (gaussian | uniform)
    'hierarchical_sample': True, # Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    'lock_view_dependence': False,
    'z_lambda': 0, # Weight for experimental latent code positional consistency loss.
    'pos_lambda': 15., # Weight parameter for experimental positional consistency loss.
    'last_back': False, # Flag to fill in background color with last sampled color on ray.
    'white_back': True,
    'use_pix_noise': False,
    'phase_noise': False,
    'delta_final': 1e10,
    'equal_lr': 1,
    'sample_lr': 1.0,
    'real_pose': True,
    'use_alpha': True,
    'alpha_delta': 0.04,
    'num_regions': 1,
    # parameters for deformation network
    'num_instances': 5
}

SPATIALSIRENBASELINEGRAM_deform_bs4split2_de1rgb1_t = {
    #FFHQ128_3dmm_face_t
    0: {'batch_size': 4, 'num_steps': 24, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'FFHQ128_3dmm_face_t'},
    int(100e3):  {'batch_size': 4, 'num_steps': 24, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4, 'dataset': 'FFHQ128_3dmm_face_t'},


    # 0: {'batch_size': 8, 'num_steps': 12, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    # # int(50): {'batch_size': 2, 'num_steps': 12, 'img_size': 256, 'batch_split': 2, 'gen_lr': 1e-6, 'disc_lr': 1e-5},
    # int(100e3): {},
    'density_lambda': 100, 
    'num_divs': 5,
    'rgb_lambda': 100,
    'sample_points': 10000,
    'points_split': 10000,
    'stage1_iters': 10000,
    'stage2_iters': 10000,
    'fov': 12, # Camera field of view
    'ray_start': 0.88, # Near clipping for camera rays.
    'ray_end': 1.12, # Far clipping for camera rays.
    'levels_start':23,
    'levels_end': 8,
    'init_radius': 0,
    'fade_steps': 10000,
    'h_stddev': 0.3, # Stddev of camera yaw in radians.
    'v_stddev': 0.155, # Stddev of camera pitch in radians.
    'h_mean': math.pi*0.5, #Mean of camera yaw in radians.
    'v_mean': math.pi*0.5, #Mean of camera pitch in radians.
    'sample_dist': 'gaussian', #Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    'topk_interval': 2000, #Interval over which to fade the top k ratio.
    'topk_v': 0.6, # Minimum fraction of a batch to keep during top k training.
    'betas': (0, 0.9), # Beta parameters for Adam.
    'unique_lr': True, # Whether to use reduced LRs for mapping network.
    'weight_decay': 0, # Weight decay parameter.
    'r1_lambda': 1, #R1 regularization parameter.
    'latent_dim': 256, #Latent dim for Siren network  in generator.
    'hidden_dim': 256, 
    'hidden_dim_sample': 128,
    'grad_clip': 0.3, #Grad clipping parameter.
    'model': 'SPATIALSIRENMULTI_NEW', # Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    'model_sample': 'SPATIALSAMPLERELU', 
    'generator': 'ImplicitGenerator3d', # Generator class. (ImplicitGenerator3d)
    'generator_module': 'generators_MPI_learn_hd',
    'discriminator': 'ProgressiveEncoderDiscriminatorAntiAlias', # Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    'clamp_mode': 'softplus', # Clamping function for Siren density output. (relu | softplus)
    'rgb_clamp_mode': 'widen_sigmoid', 
    'z_dist': 'gaussian', # Latent vector distributiion. (gaussian | uniform)
    'hierarchical_sample': True, # Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    'lock_view_dependence': False,
    'z_lambda': 0, # Weight for experimental latent code positional consistency loss.
    'pos_lambda': 15., # Weight parameter for experimental positional consistency loss.
    'last_back': False, # Flag to fill in background color with last sampled color on ray.
    'white_back': True,
    'use_pix_noise': False,
    'phase_noise': False,
    'delta_final': 1e10,
    'equal_lr': 1,
    'sample_lr': 1.0,
    'real_pose': True,
    'use_alpha': True,
    'alpha_delta': 0.04,
    'num_regions': 1,
    # parameters for deformation network
    'num_instances': 5
}