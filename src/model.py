# @title Libraries
from abc import ABC

import torch
import torch.nn as nn
from torch.autograd import Function
from src.encoder import get_encoder, CWEncoder
from src.decoder import get_decoder, CWDecoder
from src.loss import square_distance


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

    def __init__(self, encoder_name, decoder_name, cw_dim, gf, k=20, m=2048, **settings):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.encode = get_encoder(encoder_name)(cw_dim, k)
        self.decode = get_decoder(decoder_name)(cw_dim, m, gf=gf)
        self.settings = {'encode_h_dim': self.encode.h_dim, 'decode_h_dim': self.decode.h_dim, 'k': k}

    def forward(self, x):
        data = self.encoder(x)
        return self.decoder(data)

    def encoder(self, x):
        data = {'cw': self.encode(x)}
        return data

    def decoder(self, data):
        cw = data['cw']
        x = self.decode(cw).transpose(2, 1)
        data['recon'] = x
        return data

class VQVAE(AE):

    def __init__(self, encoder_name, decoder_name, cw_dim, gf, dict_size, dim_embedding, k, m):
        # encoder gives vector quantised codes, therefore the cw dim must be multiplied by the embed dim
        super().__init__(encoder_name, decoder_name, cw_dim, gf, k, m)
        self.dim_codes = cw_dim // dim_embedding
        self.dict_size = dict_size
        self.dim_embedding = dim_embedding
        self.decay_rate = 0.999
        self.dictionary = torch.nn.Parameter(
            torch.randn(self.dim_codes, self.dict_size, self.dim_embedding, requires_grad=False))
        self.ema_counts = torch.nn.Parameter(torch.ones(self.dim_codes, self.dict_size, dtype=torch.float))
        self.settings['dict_size'] = self.dict_size
        self.settings['dim_embedding'] = self.dim_embedding

    def quantise(self, mu):
        batch, embed = mu.size()
        mu2 = mu.view(batch * self.dim_codes, 1, self.dim_embedding)
        dictionary = self.dictionary.repeat(batch, 1, 1)
        dist = square_distance(mu2, dictionary)
        idx = dist.argmin(axis=2)
        cw_embed = dictionary.gather(1, idx.expand(-1, -1, self.dim_embedding))
        cw_embed = cw_embed.view(batch, self.dim_codes * self.dim_embedding)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.dict_size, device=mu.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
        # EMA update
        if self.training:
            self.ema_counts.data = self.decay_rate * self.ema_counts + (1 - self.decay_rate) * one_hot_idx.sum(0)
            mu2 = mu2 / self.ema_counts.repeat(batch, 1).unsqueeze(2).gather(1, idx).expand(-1, -1, self.dim_embedding)
            mu = mu2.view(batch, self.dim_codes, self.dim_embedding).transpose(0, 1) * (1 - self.decay_rate)
            idx = idx.view(batch, self.dim_codes, 1).transpose(0, 1).repeat(1, 1, self.dim_embedding)
            self.dictionary.data *= self.decay_rate
            self.dictionary.data.scatter_(index=idx, src=mu, dim=1, reduce='add')

        return cw_embed, one_hot_idx

    def encoder(self, x):
        data = {}
        x = self.encode(x)
        data['cw_approx'] = x
        return data


class VAECW(nn.Module):
    settings = {}

    def __init__(self, cw_dim, z_dim=20):
        super().__init__()
        self.encode = CWEncoder(cw_dim, z_dim)
        self.decode = CWDecoder(cw_dim, z_dim)
        self.settings = {'encode_h_dim': self.encode.h_dim, 'decode_h_dim': self.decode.h_dim}

    def forward(self, x):
        data = self.encoder(x)
        return self.decoder(data)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) if self.training else mu

    def encoder(self, x):
        data = {}
        x = self.encode(x)
        data['mu'], data['log_var'] = x.chunk(2, 1)
        data['z'] = self.sample(data['mu'], data['log_var'])
        return data

    def decoder(self, data):
        data['recon'] = self.decode(data['z'])
        return data


def get_model(vae, **model_settings):
    if vae == 'VAE':
        Model = AE
    elif vae == 'VQVAE':
        Model = VQVAE
    return Model(**model_settings)
