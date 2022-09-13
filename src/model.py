# @title Libraries
from abc import ABC

import torch
import torch.nn as nn
from torch.autograd import Function
from src.encoder import get_encoder
from src.decoder import get_decoder
from src.modules import LinearBlock
from src.losses import square_distance


class TransferGrad(Function):

    @staticmethod
    # transfer the grad from output to input during backprop
    def forward(ctx, input, output):
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class AE(nn.Module):
    settings = {}
    log_var = False

    def __init__(self, encoder_name, decoder_name, z_dim, gf, k=20, m=2048, **settings):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.encode = get_encoder(encoder_name)(z_dim, k, log_var=self.log_var)
        self.decode = get_decoder(decoder_name)(z_dim, m, gf=gf)
        self.settings = {'encode_h_dim': self.encode.h_dim, 'decode_h_dim': self.decode.h_dim, 'k': k}

    def forward(self, x):
        data = self.encoder(x)
        return self.decoder(data)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) if self.training else mu

    def encoder(self, x):
        data = {}
        data['z'] = self.encode(x)
        return data

    def decoder(self, data):
        z = data['z']
        x = self.decode(z).transpose(2, 1)
        data['recon'] = x
        return data

    def print_total_parameters(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total Parameters: {}'.format(num_params))
        return


class VAE(AE):
    log_var = True

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) if self.training else mu

    def encoder(self, x):
        data = {}
        x = self.encode(x)
        data['mu'], data['log_var'] = x.chunk(2, 1)
        data['z'] = self.sampling(data['mu'], data['log_var'])
        return data


class VQVAE(AE):

    def __init__(self, encoder_name, decoder_name, z_dim, in_chan, dict_size, embed_dim, k, m):
        # encoder gives vector quantised codes, therefore the z dim must be multiplied by the embed dim
        super().__init__(encoder_name, decoder_name, embed_dim * z_dim, in_chan, k, m)
        self.dim_codes = z_dim
        self.dict_size = dict_size
        self.dim_embedding = embed_dim
        self.dictionary = torch.nn.Parameter(torch.randn(self.dim_codes, self.dict_size, self.dim_embedding))
        self.settings['dict_size'] = self.dict_size
        self.settings['dim_embedding'] = self.dim_embedding

    def quantise(self, mu):
        batch, embed = mu.size()
        mu2 = mu.view(batch * self.dim_codes, 1, self.dim_embedding)
        dictionary = self.dictionary.repeat(batch, 1, 1)
        dist = square_distance(mu2, dictionary)
        idx = dist.argmin(axis=2)
        z_embed = dictionary.gather(1, idx.expand(-1, -1, self.dim_embedding))
        z_embed = z_embed.view(batch, self.dim_codes * self.dim_embedding)
        z = TransferGrad().apply(mu, z_embed)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.dict_size, device=mu.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
        return z, z_embed, one_hot_idx

    def encoder(self, x):
        data = {}
        x = self.encode(x)
        data['mu'] = x
        data['z'], data['z_embed'], data['idx'] = self.quantise(x)
        return data


class Classifier(nn.Module):
    settings = {}

    def __init__(self, encoder_name, num_classes=40, k=20):
        super().__init__()
        self.num_classes = num_classes
        self.encode = get_encoder(encoder_name)(k, task='classify')
        num_features = 2048
        self.dense = nn.Sequential(
            LinearBlock(num_features, 512),
            nn.Dropout(0.5),
            LinearBlock(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes))
        self.settings = {'encode_h_dim': self.encode.h_dim, 'dense_h_dim': [512, 256], 'k': k}

    def forward(self, x):
        features = self.encode(x)
        return {'y': self.dense(features)}


def get_model(vae, **model_settings):
    if vae == 'NoVAE':
        Model = AE
    elif vae == 'VQVAE':
        Model = VQVAE
    else:
        Model = VAE
    return Model(**model_settings)
