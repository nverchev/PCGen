import torch
import torch.nn as nn
from torch.autograd import Function
from src.encoder import get_encoder, CWEncoder
from src.decoder import get_decoder, CWDecoder
from src.loss import square_distance
from src.layer import TransferGrad

class VAECW(nn.Module):
    settings = {}

    def __init__(self, cw_dim, z_dim=64, book_size=16):
        super().__init__()
        self.z_dim = z_dim
        self.book_size = 16
        self.encoder = CWEncoder(cw_dim, z_dim)
        self.decoder = CWDecoder(cw_dim, z_dim)
        self.codebook = torch.nn.Parameter(torch.randn(1, self.book_size, z_dim))
        self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim,
                         'book_size': self.book_size}

    def forward(self, x):
        data = self.encode(x)
        return self.decode(data)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) if self.training else mu

    def quantise(self, x):
        batch, embed = x.size()
        x = x.unsqueeze(1)
        book = self.codebook.repeat(batch, 1, 1)
        dist = square_distance(x, book)
        idx = dist.argmin(axis=2)
        cw_embed = book.gather(1, idx.expand(-1, -1, self.z_dim)).squeeze(1)
        one_hot_idx = torch.zeros(batch, self.book_size, device=x.device)
        one_hot_idx = one_hot_idx.scatter_(1, idx.squeeze(-1), 1)
        return cw_embed, one_hot_idx

    def encode(self, x):
        data = {}
        x = self.encoder(x)
        data['t'], data['log_var'] = x.chunk(2, 1)
        data['t_quantised'], data['idx_t'] = self.quantise(data['t'])
        data['mu'] = data['t'] - data['t_quantised']
        data['z'] = self.sample(data['mu'], data['log_var'])
        return data

    def decode(self, data):
        data['t_recon'] = data['z'] + data['t_quantised']
        data['cw_recon'] = self.decoder(data['t_recon'])
        return data

    def reset_parameters(self):
        self.apply(lambda x: x.reset_parameters() if isinstance(x, nn.Linear) else x)


class VAECW(nn.Module):
    settings = {}

    def __init__(self, cw_dim, z_dim, codebook):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = CWEncoder(cw_dim, z_dim)
        self.decoder = CWDecoder(cw_dim, z_dim)
        self.codebook = nn.Parameter(codebook, requires_grad=False)
        self.dim_codes, self.book_size, self.dim_embedding = codebook.size()
        self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim}

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
        data['cw_recon'] = self.decoder(data['z'])
        data['cw_dist'], data['cw_recon'] = self.dist(data['cw_recon'])
        return data

    def reset_parameters(self):
        self.apply(lambda x: x.reset_parameters() if isinstance(x, nn.Linear) else x)

    def dist(self, x):
        batch, embed = x.size()
        x2 = x.view(batch * self.dim_codes, 1, self.dim_embedding)
        book = self.codebook.repeat(batch, 1, 1)
        dist = square_distance(x2, book)
        idx = dist.argmin(axis=2)
        closest = book.gather(1, idx.expand(-1, -1, self.dim_embedding)).view(batch,
                                                                              self.dim_codes * self.dim_embedding)
        return dist.sum(1).view(batch, self.dim_codes, self.book_size), closest


# class VAECW(nn.Module):
#     settings = {}
#
#     def __init__(self, cw_dim, z_dim=64, book_size=16):
#         super().__init__()
#         self.z_dim = z_dim
#         self.encoder = CWEncoder(cw_dim, z_dim)
#         self.decoder = CWDecoder(cw_dim, z_dim)
#         self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim}
#         self.dim_embedding = 4
#         self.dim_codes = z_dim // 4
#         self.book_size = book_size
#         # self.decay = 0.95
#         # self.gain = 1 - self.decay
#         self.codebook = torch.nn.Parameter(
#             torch.randn(self.dim_codes, self.book_size, self.dim_embedding)) #, requires_grad=False))
#
#     def quantise(self, x):
#         batch, embed = x.size()
#         x2 = x.contiguous().view(batch * self.dim_codes, 1, self.dim_embedding)
#         book = self.codebook.repeat(batch, 1, 1)
#         dist = square_distance(x2, book)
#         idx = dist.argmin(axis=2)
#         cw_embed = book.gather(1, idx.expand(-1, -1, self.dim_embedding))
#         cw_embed = cw_embed.view(batch, self.dim_codes * self.dim_embedding)
#         one_hot_idx = torch.zeros(batch, self.dim_codes, self.book_size, device=x.device)
#         one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
#         return cw_embed, one_hot_idx
#
#     def forward(self, x):
#         data = self.encode(x)
#         return self.decode(data)
#
#     def sample(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu) if self.training else mu
#
#     def encode(self, x):
#         data = {}
#         x = self.encoder(x)
#         data['mu'], data['log_var'] = x.chunk(2, 1)
#         data['z'] = self.sample(data['mu'], data['log_var'])
#         return data
#
#     def decode(self, data):
#         data['z_e'], data['idx_z'] = self.quantise(data['z'])
#         data['z_q'] = TransferGrad().apply(data['z'], data['z_e'])
#
#         data['cw_recon'] = self.decoder(data['z_q'])
#         return data
#
#     def reset_parameters(self):
#         self.apply(lambda x: x.reset_parameters() if isinstance(x, nn.Linear) else x)
#

class Oracle(nn.Module):
    settings = {}
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.m = 0
    def forward(self, x, indices):
        return {'recon': x[:, :self.decoder.m, :]}

class AE(nn.Module):
    settings = {}

    def __init__(self, encoder_name, decoder_name, **model_settings):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.encoder = get_encoder(encoder_name)(**model_settings)
        self.decoder = get_decoder(decoder_name)(**model_settings)
        self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim}

    def forward(self, x, indices):
        data = self.encode(x, indices)
        return self.decode(data)

    def encode(self, x, indices):
        data = {'cw': self.encoder(x, indices)}
        return data

    def decode(self, data):
        x = self.decoder(data['cw']).transpose(2, 1)
        data['recon'] = x
        return data


class VQVAE(AE):
    recon_cw = False
    transfer_grad = TransferGrad()

    def __init__(self, book_size, cw_dim,  dim_embedding, **model_settings):
        # encoder gives vector quantised codes, therefore the cw dim must be multiplied by the embed dim
        super().__init__(cw_dim=cw_dim, **model_settings)
        self.dim_codes = cw_dim // dim_embedding
        self.book_size = book_size
        self.dim_embedding = dim_embedding
        self.codebook = torch.nn.Parameter(
            torch.randn(self.dim_codes, self.book_size, self.dim_embedding))
        self.cw_encoder = VAECW(cw_dim, cw_dim // 64, self.codebook)
        self.settings['book_size'] = self.book_size
        self.settings['dim_embedding'] = self.dim_embedding
        self.settings['cw_encoder'] = self.cw_encoder.settings


    def quantise(self, x):
        batch, embed = x.size()
        x2 = x.view(batch * self.dim_codes, 1, self.dim_embedding)
        book = self.codebook.repeat(batch, 1, 1)
        dist = square_distance(x2, book)
        idx = dist.argmin(axis=2)
        cw_embed = book.gather(1, idx.expand(-1, -1, self.dim_embedding))
        cw_embed = cw_embed.view(batch, self.dim_codes * self.dim_embedding)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.book_size, device=x.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
        return cw_embed, one_hot_idx

    def encode(self, x, indices):
        data = {}
        x = self.encoder(x, indices)
        data['cw_q'] = x
        # data['cw_e'], data['idx_cw'] = self.quantise(x)
        # data['cw'] = TransferGrad().apply(x, data['cw_e'])
        return data

    def decode(self, data):
        data['cw_e'], data['cw_idx'] = self.quantise(data['cw_q'])
        data['cw'] = self.transfer_grad.apply(data['cw_q'], data['cw_e'])
        return super().decode(data)

    def forward(self, x, indices):
        if self.recon_cw:
            data = self.cw_encode(x, indices)
            return self.cw_decode(data)
        return super().forward(x, indices)

    def cw_encode(self, x, indices):
        data = self.encode(x, indices)
        data.update(self.cw_encoder.encode(data['cw_q']))
        return data

    def cw_decode(self, data):
        self.cw_encoder.decode(data)
        #data['cw_e'], data['idx_cw'] = self.quantise(data['cw_recon'])
        #data['cw'] = data['cw_e']
        # data['cw'] = TransferGrad().apply(data['cw_recon'], data['cw_e'])
        data['cw'] = data['cw_e'] = data['cw_recon']
        return super().decode(data)


def get_model(ae, **model_settings):
    model_map = dict(
                    Oracle=Oracle,
                    AE=AE,
                    VQVAE=VQVAE,
                    )
    return model_map[ae](**model_settings)
