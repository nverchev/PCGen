import numpy as np
import torch
import torch.nn as nn
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
        self.dim_codes, self.book_size, self.embedding_dim = codebook.data.size()
        self.encoder = CWEncoder(cw_dim, z_dim, embedding_dim=self.embedding_dim)
        self.decoder = CWDecoder(cw_dim, z_dim, embedding_dim=self.embedding_dim, book_size=self.book_size)
        self.n_pseudo_input = 400
        self.pseudo_inputs = nn.Parameter(torch.randn(self.n_pseudo_input, self.embedding_dim, self.dim_codes))
        self.pseudo_mu = None
        self.pseudo_log_var = None
        self.inference1 = nn.Linear(cw_dim, z_dim // 2)
        self.inference2 = nn.Sequential(nn.Linear(cw_dim + z_dim // 4, 2 * z_dim),
                                        nn.BatchNorm1d(2 * z_dim),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.Linear(2 * z_dim, 3 * z_dim // 2),
                                        )
        self.prior = nn.Sequential(nn.Linear(z_dim // 4, 2 * z_dim),
                                   nn.BatchNorm1d(2 * z_dim),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Linear(2 * z_dim, 3 * z_dim // 2),
                                   )
        self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim}

    def forward(self, x):
        data = self.encode(x)
        return self.decode(data)

    @staticmethod
    def gaussian_sample(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        if x is not None:
            x = x.view(-1, self.dim_codes, self.embedding_dim).transpose(2, 1)
            x = torch.cat([x, self.pseudo_inputs], dim=0)
        else:
            x = self.pseudo_inputs
        data = {}
        x = self.encoder(x)
        data['h'] = x[:-self.n_pseudo_input]
        x = self.inference1(x)
        data['mu'], data['log_var'] = x[:-self.n_pseudo_input].chunk(2, 1)
        data['pseudo_mu'], data['pseudo_log_var'] = x[-self.n_pseudo_input:].chunk(2, 1)
        data['z'] = self.gaussian_sample(data['mu'], data['log_var'])
        return data

    def decode(self, data):
        z1 = data['z']
        p_mu, p_logvar = self.prior(z1).chunk(2, 1)
        data['prior_log_var'] = [p_logvar]
        data['d_mu'] = []
        data['d_log_var'] = []
        if 'h' in data.keys():  # we have to inference the upper latent variable group
            d_mu, d_log_var = self.inference2(torch.cat([z1, data['h']], dim=1)).chunk(2, 1)
            z2 = self.gaussian_sample(d_mu + p_mu, d_log_var + p_logvar)
            data['d_mu'] = [d_mu]
            data['d_log_var'] = [d_log_var]
        else:
            z2 = p_mu  # self.gaussian_sample(p_mu, p_logvar)
        data['cw_recon'] = self.decoder(z2)
        data['cw_dist'], data['idx'] = self.dist(data['cw_recon'])
        return data

    def reset_parameters(self):
        self.apply(lambda x: x.reset_parameters() if isinstance(x, nn.Linear) else x)

    def dist(self, x):
        batch, embed = x.size()
        x = x.view(batch * self.dim_codes, 1, self.embedding_dim)
        book = self.codebook.detach().repeat(batch, 1, 1)
        dist = square_distance(x, book)  # Lazy vector need aggregation like sum(1) to yield tensor (|dim 1| = 1)
        idx = dist.argmin(axis=2)
        return dist.sum(1).view(batch, self.dim_codes, self.book_size), idx

    def update_pseudo_latent(self):
        pseudo_data = self.encode(None)
        self.pseudo_mu = nn.Parameter(pseudo_data['pseudo_mu'])
        self.pseudo_log_var = nn.Parameter(pseudo_data['pseudo_log_var'])


class VAECW(nn.Module):
    settings = {}

    def __init__(self, cw_dim, z_dim, codebook):
        super().__init__()
        self.z_dim = z_dim
        self.codebook = codebook
        self.dim_codes, self.book_size, self.embedding_dim = codebook.data.size()
        self.encoder = CWEncoder(cw_dim, z_dim, embedding_dim=self.embedding_dim)
        self.decoder = CWDecoder(cw_dim, z_dim, embedding_dim=self.embedding_dim, book_size=self.book_size)
        self.n_pseudo_input = 400
        self.pseudo_inputs = nn.Parameter(torch.randn(self.n_pseudo_input, self.embedding_dim, self.dim_codes))
        self.pseudo_mu = None
        self.pseudo_log_var = None
        self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim}

    def forward(self, x):
        data = self.encode(x)
        return self.decode(data)

    @staticmethod
    def gaussian_sample(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        if x is not None:
            x = x.view(-1, self.dim_codes, self.embedding_dim).transpose(2, 1)
            x = torch.cat([x, self.pseudo_inputs], dim=0)
        else:
            x = self.pseudo_inputs
        data = {}
        x = self.encoder(x)
        data['mu'], data['log_var'] = x[:-self.n_pseudo_input].chunk(2, 1)
        data['pseudo_mu'], data['pseudo_log_var'] = x[-self.n_pseudo_input:].chunk(2, 1)
        data['z'] = self.gaussian_sample(data['mu'], data['log_var'])

        return data

    def decode(self, data):
        z1 = data['z']
        data['prior_log_var'] = []
        data['d_mu'] = []
        data['d_log_var'] = []
        data['cw_recon'] = self.decoder(z1)
        data['cw_dist'], data['idx'] = self.dist(data['cw_recon'])
        return data

    def reset_parameters(self):
        self.apply(lambda x: x.reset_parameters() if isinstance(x, nn.Linear) else x)

    def dist(self, x):
        batch, embed = x.size()
        x = x.view(batch * self.dim_codes, 1, self.embedding_dim)
        book = self.codebook.detach().repeat(batch, 1, 1)
        dist = square_distance(x, book)  # Lazy vector need aggregation like sum(1) to yield tensor (|dim 1| = 1)
        idx = dist.argmin(axis=2)
        return dist.sum(1).view(batch, self.dim_codes, self.book_size), idx

    def update_pseudo_latent(self):
        pseudo_data = self.encode(None)
        self.pseudo_mu = nn.Parameter(pseudo_data['pseudo_mu'])
        self.pseudo_log_var = nn.Parameter(pseudo_data['pseudo_log_var'])


class BaseModel(nn.Module):
    settings = {}

    def __init__(self, m_training, m_test, **kwargs):
        super().__init__()
        self.m_training = m_training
        self.m_test = m_test

    @property
    def m(self):
        return self.m_training if self.training else self.m_test

    def forward(self, x, indices):
        return NotImplementedError


class Oracle(BaseModel):

    def forward(self, x, indices):
        return {'recon': x[:, :self.m, :]}


class AE(BaseModel):
    settings = {}

    def __init__(self, encoder_name, decoder_name, **args):
        super().__init__(**args)
        self.encoder = get_encoder(encoder_name)(**args)
        self.decoder = get_decoder(decoder_name)(**args)
        self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim}

    def forward(self, x, indices):
        data = self.encode(x, indices)
        return self.decode(data)

    def encode(self, x, indices):
        data = {'cw': self.encoder(x, indices)}
        return data

    def decode(self, data):
        x = self.decoder(data['cw'], self.m).transpose(2, 1)
        data['recon'] = x.contiguous()
        return data


class VQVAE(AE):

    def __init__(self, book_size, cw_dim, z_dim, embedding_dim, vq_ema_update, **model_settings):
        # encoder gives vector quantised codes, therefore the cw dim must be multiplied by the embed dim
        super().__init__(cw_dim=cw_dim, **model_settings)
        self.double_encoding = False
        self.dim_codes = cw_dim // embedding_dim
        self.book_size = book_size
        self.embedding_dim = embedding_dim
        self.vq_ema_update = vq_ema_update
        self.codebook = torch.nn.Parameter(
            torch.randn(self.dim_codes, self.book_size, self.embedding_dim, requires_grad=not vq_ema_update))
        if vq_ema_update:
            self.decay = .999
            self.gain = 1 - self.decay
            self.ema_counts = torch.nn.Parameter(
                torch.ones(self.dim_codes, self.book_size, dtype=torch.float, requires_grad=False))
        self.cw_encoder = VAECW(cw_dim, z_dim, self.codebook)
        self.settings['book_size'] = self.book_size
        self.settings['embedding_dim'] = self.embedding_dim
        self.settings['cw_encoder'] = self.cw_encoder.settings

    def quantise(self, x):
        batch, embed = x.size()
        x = x.view(batch * self.dim_codes, 1, self.embedding_dim)
        book = self.codebook.repeat(batch, 1, 1)
        dist = square_distance(x, book)
        idx = dist.argmin(axis=2)
        cw_embed = self.get_quantised_code(idx, book)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.book_size, device=x.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
        # EMA update
        if self.training and self.vq_ema_update:
            x = x.view(batch, self.dim_codes, self.embedding_dim).transpose(0, 1)
            idx = idx.view(batch, self.dim_codes, 1).transpose(0, 1).expand(-1, -1, self.embedding_dim)
            update_dict = torch.zeros_like(self.codebook).scatter_reduce_(index=idx, src=x, dim=1, reduce='sum')
            normalize = self.ema_counts.unsqueeze(2).expand(-1, -1, self.embedding_dim)
            self.codebook.data = self.codebook * self.decay + self.gain * update_dict / (normalize + 1e-6)
            self.ema_counts.data = self.decay * self.ema_counts + self.gain * one_hot_idx.sum(0)

        return cw_embed, one_hot_idx

    def forward(self, x, indices):
        data = self.encode(x, indices, cw_encoding=self.double_encoding)
        return self.decode(data)

    def encode(self, x, indices, cw_encoding=False):
        data = {'cw_q': self.encoder(x, indices)}
        if cw_encoding:
            data.update(self.cw_encoder.encode(x.detach()))
        return data

    def decode(self, data, from_z=False):
        if from_z:
            self.cw_encoder.decode(data)  # looks for the z keyword
            idx = data['idx']
            data['cw'] = self.get_quantised_code(idx, self.codebook.repeat(idx.shape[0] // self.dim_codes, 1, 1))
        else:
            data['cw_e'], data['cw_idx'] = self.quantise(data['cw_q'])
            data['cw'] = TransferGrad().apply(data['cw_q'], data['cw_e'])  # call class method, do not instantiate
        return super().decode(data)

    @torch.inference_mode()
    def random_sampling(self, batch_size):
        self.eval()
        if self.cw_encoder.pseudo_mu is None:
            self.cw_encoder.update_pseudo_latent()
        pseudo_mu = self.cw_encoder.pseudo_mu
        pseudo_log_var = self.cw_encoder.pseudo_log_var
        pseudo_z = []
        for _ in range(batch_size):
            i = np.random.randint(self.cw_encoder.n_pseudo_input)
            pseudo_z.append(self.cw_encoder.gaussian_sample(pseudo_mu[i], pseudo_log_var[i]))
        pseudo_z = torch.stack(pseudo_z)
        data = {'z': pseudo_z.contiguous()}
        out = self.decode(data=data, from_z=True)
        return out

    def get_quantised_code(self, idx, book):
        idx = idx.expand(-1, -1, self.embedding_dim)
        return book.gather(1, idx).view(-1, self.dim_codes * self.embedding_dim)


def get_model(model_head, **model_settings):
    model_map = dict(
        Oracle=Oracle,
        AE=AE,
        VQVAE=VQVAE,
    )
    return model_map[model_head](**model_settings)
