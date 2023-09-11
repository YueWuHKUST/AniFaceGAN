import os
from tkinter import N

from matplotlib.pyplot import axis

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import torchvision
import glob
import PIL
import random
import math
import numpy as np
import zipfile
import time
from scipy.io import loadmat
import meshio

def read_pose_ori(name,flip=False):
    P = loadmat(name)['angle']
    P_x = -(P[0,0] - 0.1) + math.pi/2
    if not flip:
        P_y = P[0,1] + math.pi/2
    else:
        P_y = -P[0,1] + math.pi/2


    P = torch.tensor([P_x,P_y],dtype=torch.float32)

    return P
    
def read_pose(mat_flile, flip=False):
    P = mat_flile['angle']
    P_x = -(P[0,0] - 0.1) + math.pi/2
    if not flip:
        P_y = P[0,1] + math.pi/2
    else:
        P_y = -P[0,1] + math.pi/2
    P = torch.tensor([P_x,P_y],dtype=torch.float32)
    return P

def read_latents(mat_file=None):
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
    latents = mat_file
    latent_id = torch.from_numpy(latents['id']).float()[0,...]
    latent_exp = torch.from_numpy(latents['exp']).float()[0,...]
    return latent_id, latent_exp




def read_pose_npy(name,flip=False):
    P = np.load(name)
    P_x = P[0] + 0.14
    if not flip:
        P_y = P[1]
    else:
        P_y = -P[1] + math.pi


    P = torch.tensor([P_x,P_y],dtype=torch.float32)

    return P


def transform_matrix_to_camera_pos(c2w,flip=False):
    """
    Get camera position with transform matrix

    :param c2w: camera to world transform matrix
    :return: camera position on spherical coord
    """

    c2w[[0,1,2]] = c2w[[1,2,0]]
    pos = c2w[:, -1].squeeze()
    radius = float(np.linalg.norm(pos))
    theta = float(np.arctan2(-pos[0], pos[2]))
    phi = float(np.arctan(-pos[1] / np.linalg.norm(pos[::2])))
    theta = theta + np.pi * 0.5
    phi = phi + np.pi * 0.5
    if flip:
        theta = -theta + math.pi
    P = torch.tensor([phi,theta],dtype=torch.float32)
    return P


class FFHQ128(Dataset):
    def __init__(self, opt, img_size, **kwargs):
        super().__init__()
        num_files = 69994 if opt.debug_mode == False else 30
        self.data = sorted(glob.glob(os.path.join('../datasets/image256_align_new_mirror_wo_t','*.png')))[:num_files]
        self.pose = [os.path.join('../datasets/ffhq_pose_align_new_mirror',f.split('/')[-1].replace('png','mat')) for f in self.data][:num_files]
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.Resize((img_size, img_size), interpolation=2)])
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)
        flip = (torch.rand(1) < 0.5)
        if flip:
            X = F.hflip(X)
        P = read_pose_ori(self.pose[index],flip=flip)
        return X, P




class ImageDatasetZip(Dataset):
    def __init__(self, img_size,zip_name,max_num=1000, **kwargs):
        super().__init__()

        self._path = zip_name
        self._zipfile = None
        self._all_fnames = set(self._get_zipfile().namelist())

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).size)
        self._raw_shape = list(raw_shape)
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if max_num is not None:
            self._raw_idx = self._raw_idx[:max_num]

        self.num_images = len(self._image_fnames)

        self.all_images = []
        for i in range(self._raw_idx.size):
            print(i)
            img = self._load_raw_image(self._raw_idx[i])
            self.all_images.append(img)

        self.transform = transforms.Compose(
                    [transforms.Resize(320), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((img_size, img_size), interpolation=0)])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        return self._get_zipfile().open(fname, 'r')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image = PIL.Image.open(f).copy()
        return image


    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, index):
        # X = self._load_raw_image(self._raw_idx[index])
        X = self.all_images[index]
        # X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


def get_dataset(dataset, batch_size=1):#:, **kwargs):
    #dataset = globals()[name](opt, **metadata, **kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3

def get_dataset_distributed_(_dataset, world_size, rank, n_workers, batch_size, **kwargs):

    sampler = torch.utils.data.distributed.DistributedSampler(
        _dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        _dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=n_workers,
        # prefetch_factor=batch_size, 
    )

    return dataloader, 3