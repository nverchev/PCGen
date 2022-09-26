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
        self.encoder = get_encoder(encoder_name)(cw_dim, k)
        self.decoder = get_decoder(decoder_name)(cw_dim, m, gf=gf)
        self.settings = {'encode_h_dim': self.encode.h_dim, 'decode_h_dim': self.decode.h_dim, 'k': k}

    def forward(self, x):
        data = self.encode(x)
        return self.decode(data)

    def encode(self, x):
        data = {'cw': self.encoder(x)}
        return data

    def decode(self, data):
        x = self.decoder(data['cw']).transpose(2, 1)
        data['recon'] = x
        return data


class VAECW(nn.Module):
    settings = {}

    def __init__(self, cw_dim, z_dim=20):
        super().__init__()
        self.encoder = CWEncoder(cw_dim, z_dim)
        self.decoder = CWDecoder(cw_dim, z_dim)
        # self.settings = {'encode_h_dim': self.encode.h_dim, 'decode_h_dim': self.decode.h_dim}

    def forward(self, x):
        data = self.encode(x)
        return self.decode(data)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) if self.training else mu

    def encode(self, x):
        data = {}
        x = self.encoder(x)
        data['mu'], data['log_var'] = x.chunk(2, 1)
        data['z'] = self.sample(data['mu'], data['log_var'])
        return data

    def decode(self, data):
        data['recon_cw'] = self.decoder(data['z'])
        return data


class VQVAE(AE):

    def __init__(self, encoder_name, decoder_name, cw_dim, gf, dict_size, dim_embedding, k, m):
        # encoder gives vector quantised codes, therefore the cw dim must be multiplied by the embed dim
        super().__init__(encoder_name, decoder_name, cw_dim, gf, k, m)
        self.recon_z = False
        self.dim_codes = cw_dim // dim_embedding
        self.dict_size = dict_size
        self.dim_embedding = dim_embedding
        self.decay_rate = 0.999
        self.dictionary = torch.nn.Parameter(
            torch.randn(self.dim_codes, self.dict_size, self.dim_embedding, requires_grad=False))
        self.ema_counts = torch.nn.Parameter(torch.ones(self.dim_codes, self.dict_size, dtype=torch.float))
        self.cw_encoder = VAECW(cw_dim, cw_dim // 64)
        self.settings['dict_size'] = self.dict_size
        self.settings['dim_embedding'] = self.dim_embedding
        self.settings['cw_encoder'] = self.cw_encoder.settings

    def quantise(self, x):
        batch, embed = x.size()
        x2 = x.view(batch * self.dim_codes, 1, self.dim_embedding)
        dictionary = self.dictionary.repeat(batch, 1, 1)
        dist = square_distance(x2, dictionary)
        idx = dist.argmin(axis=2)
        cw_embed = dictionary.gather(1, idx.expand(-1, -1, self.dim_embedding))
        cw_embed = cw_embed.view(batch, self.dim_codes * self.dim_embedding)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.dict_size, device=x.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
        # EMA update
        if self.training:
            self.ema_counts.data = self.decay_rate * self.ema_counts + (1 - self.decay_rate) * one_hot_idx.sum(0)
            x2 = x2 / self.ema_counts.repeat(batch, 1).unsqueeze(2).gather(1, idx).expand(-1, -1, self.dim_embedding)
            x = x2.view(batch, self.dim_codes, self.dim_embedding).transpose(0, 1) * (1 - self.decay_rate)
            idx = idx.view(batch, self.dim_codes, 1).transpose(0, 1).repeat(1, 1, self.dim_embedding)
            self.dictionary.data *= self.decay_rate
            self.dictionary.data.scatter_(index=idx, src=x, dim=1, reduce='add')

        return cw_embed, one_hot_idx

    def encode(self, x):
        data = {}
        x = self.encoder(x)
        data['cw_approx'] = x
        data['cw_embed'], data['idx'] = self.quantise(x)
        data['cw'] = TransferGrad().apply(x, data['cw_embed'])
        return data

    def forward(self, x):
        if self.recon_z:
            data = self.encode_z(x)
            return self.decode_z(data)
        return super().forward(x)

    def encode_z(self, x):
        data = self.encode(x)
        data.update(self.cw_encoder.encode(data['cw_approx']))
        return data

    def decode_z(self, data):
        data.update(self.cw_encoder.encode(data['z']))
        x = self.decoder(data['recon_cw']).transpose(2, 1)
        data['recon'] = x
        return data


def get_model(vae, **model_settings):
    if vae == 'AE':
        Model = AE
    elif vae == 'VQVAE':
        Model = VQVAE
    return Model(**model_settings)
