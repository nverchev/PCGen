# @title Libraries
from abc import ABC

import torch
import torch.nn as nn
from src.encoder import get_encoder
from src.decoder import get_decoder
from src.modules import LinearBlock
from src.utils import square_distance
from torch.autograd import Function


class TransferGrad(Function, ABC):

    @staticmethod
    # transfer the grad from output to input during backprop
    def forward(ctx, input, output):
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class VAE(nn.Module):
    settings = {}
    vq = False

    def __init__(self, encoder_name, decoder_name, z_dim, in_chan, k, m):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.encode = get_encoder(encoder_name)(in_chan, z_dim, k, vq=self.vq)
        self.decode = get_decoder(decoder_name)(z_dim, m)
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
        x = self.encode(x)
        data['mu'], data['log_var'] = x.chunk(2, 1)
        data['z'] = self.sampling(data['mu'], data['log_var'])
        return data

    def decoder(self, data):
        z = data['z']
        x = self.decode(z)
        data['recon'] = x
        return data

    def print_total_parameters(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total Parameters: {}'.format(num_params))
        return


class VQVAE(VAE):
    settings = {}
    vq = True
    def __init__(self, encoder_name, decoder_name, z_dim, in_chan, k, m):
        super().__init__(encoder_name, decoder_name, z_dim, in_chan, k, m)



    def quantise(self, mu):
        batch, embed = mu.size()
        mu2 = mu.view(batch * self.dim_codes, 1, self.dim_embedding)
        dict = self.dictionary.repeat(batch, 1, 1)
        dist = square_distance(mu2, dict)
        idx = dist.argmin(axis=2)
        z_embed = dict.gather(1, idx.expand(-1, -1, self.dim_embedding)).view(batch,
                                                                              self.dim_codes * self.dim_embedding)
        z = TransferGrad().apply(mu, z_embed)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.dict_size, device=mu.device).scatter_(2,
                                                                                                    idx.view(batch,
                                                                                                             self.dim_codes,
                                                                                                             1), 1)
        return z, z_embed, one_hot_idx

    def encoder(self, x):
        data = {}
        x = self.encode(x)
        data['mu'] = x
        data['z'] = self.quantise(x)
        return data


def get_model(encoder_name, decoder_name, z_dim, in_chan, k=20, m=2048, vector_quantised=False):
    Model = VQVAE if vector_quantised else VAE
    return Model(encoder_name, decoder_name, z_dim, in_chan, k, m)


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
