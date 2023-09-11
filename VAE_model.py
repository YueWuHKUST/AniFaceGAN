from base64 import encode
from math import gamma
import torch 
import torch.nn as nn


import torch
from torch import nn
from torch.nn import functional as F


class VAE_ID(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dim=512, hidden_nums=5, **kwargs) -> None:
        super(VAE_ID, self).__init__()
        '''
        in_channels: the input channel of coeff
        latent_dim: the mapped gaussian distribution
        '''
        self.epoch = 0
        self.step = 0

        self.latent_dim = latent_dim
        self.in_channels_ori = in_channels
        modules = []

        # Build Encoder
        for _ in range(hidden_nums):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.LeakyReLU())
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        for _ in range(hidden_nums):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_dim, self.in_channels_ori)
                            )
        for param in self.parameters():
            param.requires_grad = False

    def set_device(self, device):
        self.device = device
        self.encoder.device = device
        self.fc_mu.device = device
        self.fc_var.device = device
        self.decoder.device = device
        self.final_layer.device = device


    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        #print("result shape", result.size())
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)# 4 x 128 
        #print("mu shape", mu.size())
        log_var = self.fc_var(result) # 4 x 128
        #print("log var shape", log_var.size())

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        #print("mu", mu.size(), "logvar", log_var.size(), "z", z.size())
        recons = self.decode(z)
        #print("recons", recons.size())
        return  [recons, input, mu, log_var]

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

class VAE_EXP(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dim=512, hidden_nums=5, **kwargs) -> None:
        super(VAE_EXP, self).__init__()
        '''
        in_channels: the input channel of coeff
        latent_dim: the mapped gaussian distribution
        '''
        self.epoch = 0
        self.step = 0

        self.latent_dim = latent_dim
        self.in_channels_ori = in_channels
        modules = []

        # Build Encoder
        for _ in range(hidden_nums):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.LeakyReLU())
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        for _ in range(hidden_nums):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_dim, self.in_channels_ori)
                            )
        for param in self.parameters():
            param.requires_grad = False

    def set_device(self, device):
        self.device = device
        self.encoder.device = device
        self.fc_mu.device = device
        self.fc_var.device = device
        self.decoder.device = device
        self.final_layer.device = device


    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        #print("result shape", result.size())
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)# 4 x 128 
        #print("mu shape", mu.size())
        log_var = self.fc_var(result) # 4 x 128
        #print("log var shape", log_var.size())

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        #print("mu", mu.size(), "logvar", log_var.size(), "z", z.size())
        recons = self.decode(z)
        #print("recons", recons.size())
        return  [recons, input, mu, log_var]

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]