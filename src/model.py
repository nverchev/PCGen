# @title Libraries
from abc import ABC

import torch
import torch.nn as nn
from torch.autograd import Function
from src.encoder import get_encoder
from src.decoder import get_decoder
from src.layer import LinearLayer
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
    log_var = False

    def __init__(self, encoder_name, decoder_name, cw_dim, gf, k=20, m=2048, **settings):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.encode = get_encoder(encoder_name)(cw_dim, k, log_var=self.log_var)
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

    def print_total_parameters(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total Parameters: {}'.format(num_params))
        return


class VAE(AE):
    log_var = True

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) if self.training else mu

    def encoder(self, x):
        data = {}
        x = self.encode(x)
        data['mu'], data['log_var'] = x.chunk(2, 1)
        data['cw'] = self.sample(data['mu'], data['log_var'])
        return data


class VQVAE(VAE):
    log_var = False

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
        self.encode_cw = nn.Sequential(LinearLayer(cw_dim, cw_dim // 4, act=None),
                                       LinearLayer(cw_dim // 4, cw_dim // 16),
                                       LinearLayer(cw_dim // 16, cw_dim // 32))
        self.decode_cw = nn.Sequential(LinearLayer(cw_dim // 64, cw_dim // 16),
                                       LinearLayer(cw_dim // 16, cw_dim // 4),
                                       LinearLayer(cw_dim // 4, cw_dim, act=None))

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
        data['mu'], data['log_var'] = self.encode_cw(x).chunk(2, 1)
        data['z'] = self.sample(data['mu'], data['log_var'])
        return data

    def decoder(self, data):
        data['cw_recon'] = self.decode_cw(data['z'])
        if self.training and 'cw_approx' in data:
            data['cw_embed'], data['idx'] = self.quantise(data['cw_approx'])
            data['cw'] = TransferGrad().apply(data['cw_approx'], data['cw_embed'])
        else:
            data['cw_embed'], data['idx'] = self.quantise(data['cw_recon'])
            data['cw'] = data['cw_embed']
        x = self.decode(data['cw']).transpose(2, 1)
        data['recon'] = x
        return data


class Classifier(nn.Module):
    settings = {}

    def __init__(self, encoder_name, num_classes=40, k=20):
        super().__init__()
        self.num_classes = num_classes
        self.encode = get_encoder(encoder_name)(k, task='classify')
        num_features = 2048
        self.dense = nn.Sequential(
            LinearLayer(num_features, 512),
            nn.Dropout(0.5),
            LinearLayer(512, 256),
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
