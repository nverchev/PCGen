import numpy as np
import torch
import torch.nn as nn
from src.trainer import UsuallyFalse
from torch.autograd import Function
from src.encoder import get_encoder, CWEncoder
from src.decoder import get_decoder, CWDecoder
from src.loss_and_metrics import square_distance
from src.layer import TransferGrad


class VAECW(nn.Module):
    settings = {}

    def __init__(self, cw_dim, z_dim, codebook):
        super().__init__()
        self.z_dim = z_dim
        self.codebook = codebook
        self.dim_codes, self.book_size, self.dim_embedding = codebook.data.size()
        self.encoder = CWEncoder(cw_dim, z_dim, dim_embedding=self.dim_embedding)
        self.decoder = CWDecoder(cw_dim, z_dim, dim_embedding=self.dim_embedding, book_size=self.book_size)
        self.inference1 = nn.Linear(2 * z_dim, z_dim // 2)
        self.inference2 = nn.Sequential(nn.Linear(9 * z_dim // 4, 2 * z_dim),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Linear(2 * z_dim, 3 * z_dim // 2),
                                        )
        self.prior = nn.Sequential(nn.Linear(z_dim // 4, 2 * z_dim),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Linear(2 * z_dim, 3 * z_dim // 2),
                                   )
        self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim}

    def forward(self, x):
        data = self.encode(x)
        return self.decode(data)

    def gaussian_sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) if self.training else mu

    def encode(self, x):
        data = {}
        data['h'] = self.encoder(x.view(-1, self.dim_codes, self.dim_embedding).transpose(2, 1))
        mu, log_var = self.inference1(data['h']).chunk(2, 1)
        data['mu'], data['log_var'] = mu, log_var
        data['z'] = [self.gaussian_sample(data['mu'], data['log_var'])]
        return data

    def decode(self, data):
        z1 = data['z'][0]
        p_mu, p_logvar = self.prior(z1).chunk(2, 1)
        data['prior_log_var'] = [p_logvar]
        if 'h' in data.keys():  # we have to inference the upper latent variable group
            d_mu, d_log_var = self.inference2(torch.cat([z1, data['h']], dim=1)).chunk(2, 1)
            z2 = self.gaussian_sample(d_mu + p_mu, d_log_var + p_logvar)
            data['d_mu'] = [d_mu]
            data['d_log_var'] = [d_log_var]
        else:
            z2 = self.gaussian_sample(p_mu, p_logvar)
        data['cw_recon'] = self.decoder(z2)
        data['cw_dist'], data['idx'] = self.dist(data['cw_recon'])
        return data

    #
    # def reset_parameters(self):
    #     self.apply(lambda x: x.reset_parameters() if isinstance(x, nn.Linear) else x)

    def dist(self, x):
        batch, embed = x.size()
        x = x.view(batch * self.dim_codes, 1, self.dim_embedding)
        book = self.codebook.detach().repeat(batch, 1, 1)
        dist = square_distance(x, book)  # Lazy vector need aggregation like sum(1) to yield tensor (|dim 1| = 1)
        idx = dist.argmin(axis=2)
        return dist.sum(1).view(batch, self.dim_codes, self.book_size), idx



class Oracle(nn.Module):
    settings = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.m = np.Inf

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
    transfer_grad = TransferGrad()
    from_z = UsuallyFalse()

    def __init__(self, book_size, cw_dim, z_dim, dim_embedding, **model_settings):
        # encoder gives vector quantised codes, therefore the cw dim must be multiplied by the embed dim
        super().__init__(cw_dim=cw_dim, **model_settings)
        self.dim_codes = cw_dim // dim_embedding
        self.book_size = book_size
        self.dim_embedding = dim_embedding
        # self.decay = .99
        # self.gain = 1 - self.decay
        self.codebook = torch.nn.Parameter(
            torch.randn(self.dim_codes, self.book_size, self.dim_embedding))  # , requires_grad=False))
        # self.ema_counts = torch.nn.Parameter(
        #     torch.ones(self.dim_codes, self.book_size, dtype=torch.float, requires_grad=False))
        self.cw_encoder = VAECW(cw_dim, z_dim, self.codebook)
        self.settings['book_size'] = self.book_size
        self.settings['dim_embedding'] = self.dim_embedding
        self.settings['cw_encoder'] = self.cw_encoder.settings

    def quantise(self, x):
        batch, embed = x.size()
        x = x.view(batch * self.dim_codes, 1, self.dim_embedding)
        book = self.codebook.repeat(batch, 1, 1)
        dist = square_distance(x, book)
        idx = dist.argmin(axis=2)
        cw_embed = self.get_quantised_code(idx, book)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.book_size, device=x.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
        # # EMA update
        # if self.training:
        #     x = x.view(batch, self.dim_codes, self.dim_embedding).transpose(0, 1)
        #     idx = idx.view(batch, self.dim_codes, 1).transpose(0, 1).expand(-1, -1, self.dim_embedding)
        #     update_dict = torch.zeros_like(self.codebook).scatter_(index=idx, src=x, dim=1, reduce='add')
        #     normalize = self.ema_counts.unsqueeze(2).expand(-1, -1, self.dim_embedding)
        #     self.codebook.data = self.codebook * self.decay + self.gain * update_dict / (normalize + 1e-6)
        #     self.ema_counts.data = self.decay * self.ema_counts + self.gain * one_hot_idx.sum(0)

        return cw_embed, one_hot_idx

    def encode(self, x, indices):
        data = {}
        x = self.encoder(x, indices)
        data['cw_q'] = x
        data.update(self.cw_encoder.encode(x if self.from_z else x.detach()))
        return data

    # def decode(self, data):
    #     self.cw_encoder.decode(data)
    #     idx = data['idx']
    #     data['cw'] = self.get_quantised_code(idx, self.codebook.repeat(idx.shape[0] // self.dim_codes, 1, 1))
    #     data['cw_e'], data['cw_idx'] = self.quantise(data['cw_q'])
    #     return super().decode(data)

    def decode(self, data):
        self.cw_encoder.decode(data)
        if self.from_z:
            idx = data['idx']
            data['cw'] = self.get_quantised_code(idx, self.codebook.repeat(idx.shape[0] // self.dim_codes, 1, 1))
            if 'cw_q' in list(data):
                data['cw_e'], data['cw_idx'] = self.quantise(data['cw_q'])
        else:
            data['cw_e'], data['cw_idx'] = self.quantise(data['cw_q'])
            data['cw'] = self.transfer_grad.apply(data['cw_q'], data['cw_e'])
            self.cw_encoder.decode(data)

        return super().decode(data)

    def random_sampling(self, batch_size):
        torch.random.seed()
        data = {'z': [torch.randn(batch_size, self.cw_encoder.z_dim // 4).to(self.codebook.device)]}
        with self.from_z:
            out = self.decode(data=data)
        return out

    def get_quantised_code(self, idx, book):
        idx = idx.expand(-1, -1, self.dim_embedding)
        return book.gather(1, idx).view(-1, self.dim_codes * self.dim_embedding)


def get_model(ae, **model_settings):
    model_map = dict(
        Oracle=Oracle,
        AE=AE,
        VQVAE=VQVAE,
    )
    return model_map[ae](**model_settings)
