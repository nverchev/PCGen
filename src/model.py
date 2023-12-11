import numpy as np
import torch
import torch.nn as nn
from src.encoder import get_encoder, WEncoder
from src.decoder import get_decoder, WDecoder
from src.neighbour_op import square_distance
from src.layer import TransferGrad
from src.utils import UsuallyFalse


class VAECW(nn.Module):
    settings = {}

    def __init__(self, w_dim, z_dim, codebook, vae_n_pseudo_inputs, vae_dropout, **_):
        super().__init__()
        self.z_dim = z_dim
        self.codebook = codebook
        self.dim_codes, self.book_size, self.embedding_dim = codebook.data.size()
        self.encoder = WEncoder(w_dim, z_dim, self.embedding_dim, vae_dropout)
        self.decoder = WDecoder(w_dim, z_dim, self.embedding_dim, self.book_size, vae_dropout)
        self.n_pseudo_input = vae_n_pseudo_inputs
        self.pseudo_inputs = nn.Parameter(torch.randn(self.n_pseudo_input, self.embedding_dim, self.dim_codes))
        self.pseudo_mu = nn.Parameter(torch.empty(self.n_pseudo_input, z_dim))
        self.pseudo_log_var = nn.Parameter(torch.empty(self.n_pseudo_input, z_dim))
        self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim}

    def forward(self, x, data=None):
        data = data if data is not None else {}
        data.update(self.encode(x))
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
        self.pseudo_mu.data = data['pseudo_mu']
        self.pseudo_log_var.data = data['pseudo_log_var']
        return data

    def decode(self, data):
        z = data['z']
        data['cw_recon'] = self.decoder(z)
        data['cw_dist'], data['idx'] = self.dist(data['cw_recon'])
        return data

    def recursive_reset_parameters(self):
        self.apply(lambda x: x.reset_parameters() if hasattr(x, 'reset_parameters') else x)

    def dist(self, x):
        batch, _ = x.shape
        x = x.view(batch * self.dim_codes, 1, self.embedding_dim)
        book = self.codebook.detach().repeat(batch, 1, 1)
        dist = square_distance(x, book)  # Lazy vector need aggregation like sum(1) to yield tensor (|dim 1| = 1)
        idx = dist.argmin(axis=2)
        return dist.sum(1).view(batch, self.dim_codes, self.book_size), idx.view(batch, self.dim_codes, 1)

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

    def recursive_reset_parameters(self):
        self.apply(lambda x: x.reset_parameters() if hasattr(x, 'reset_parameters') else x)


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

    def forward(self, x, indices, initial_sampling=None, viz_att=None, viz_components=None):
        data = self.encode(x, indices)
        return self.decode(data, initial_sampling, viz_att, viz_components)

    def encode(self, x, indices):
        data = {'cw': self.encoder(x, indices)}
        return data

    def decode(self, data, initial_sampling=None, viz_att=None, viz_components=None):
        additional_arguments = []
        if self.decoder.__class__.__name__ == 'PCGen':
            additional_arguments.extend([initial_sampling, viz_att, viz_components])
        x = self.decoder(data['cw'], self.m, *additional_arguments)
        data['recon'] = x.transpose(2, 1).contiguous()
        return data


class VQVAE(AE):

    def __init__(self, book_size, w_dim, z_dim, embedding_dim, vq_ema_update, **model_settings):
        # encoder gives vector quantised codes, therefore the cw dim must be multiplied by the embed dim
        super().__init__(w_dim=w_dim, **model_settings)
        self.double_encoding = UsuallyFalse()
        self.dim_codes = w_dim // embedding_dim
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
        self.cw_encoder = VAECW(w_dim, z_dim, self.codebook, **model_settings)
        self.settings['book_size'] = self.book_size
        self.settings['embedding_dim'] = self.embedding_dim
        self.settings['cw_encoder'] = self.cw_encoder.settings

    def quantise(self, x):
        batch, embed = x.size()
        x = x.view(batch * self.dim_codes, 1, self.embedding_dim)
        book = self.codebook.repeat(batch, 1, 1)
        dist = square_distance(x, book)
        idx = dist.argmin(axis=2).view(-1, 1, 1)
        cw_embed = self.get_quantised_code(idx, book)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.book_size, device=x.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
        # EMA update
        if self.training and self.vq_ema_update:
            x = x.view(batch, self.dim_codes, self.embedding_dim).transpose(0, 1)
            idx = idx.view(batch, self.dim_codes, 1).transpose(0, 1).expand(-1, -1, self.embedding_dim)
            update_dict = torch.zeros_like(self.codebook).scatter_(index=idx, src=x, dim=1, reduce='sum')
            normalize = self.ema_counts.unsqueeze(2).expand(-1, -1, self.embedding_dim)
            self.codebook.data = self.codebook * self.decay + self.gain * update_dict / (normalize + 1e-6)
            self.ema_counts.data = self.decay * self.ema_counts + self.gain * one_hot_idx.sum(0)

        return cw_embed, one_hot_idx

    def encode(self, x, indices):
        data = {'cw_q': self.encoder(x, indices)}
        data.update(self.cw_encoder.encode(data['cw_q'].detach()))
        return data

    def decode(self, data, initial_sampling=None, viz_att=None, viz_components=None):
        if self.double_encoding:
            self.cw_encoder.decode(data)  # looks for the z keyword
            idx = data['idx']
            batch = idx.shape[0]
            book = self.codebook.repeat(batch, 1, 1)
            # TODO Remove this when sampling
            one_hot_idx = torch.zeros(batch, self.dim_codes, self.book_size, device=idx.device)
            data['one_hot_idx'] = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
            data['cw_e'] = data['cw'] = self.get_quantised_code(idx.view(batch * idx.shape[1], 1, 1), book)
        else:
            data['cw_e'], data['one_hot_idx'] = self.quantise(data['cw_q'])
            data['cw'] = TransferGrad().apply(data['cw_q'], data['cw_e'])  # call class method, do not instantiate
            # FIXME decide if keeping this or not
            self.cw_encoder.decode(data)  # looks for the z keyword
        return super().decode(data, initial_sampling, viz_att, viz_components)

    @torch.inference_mode()
    def random_sampling(self, batch_size, initial_sampling=None, viz_att=None, viz_components=None, z_bias=0):
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
        # pseudo_z = torch.from_numpy(self.gm.sample(batch_size)[0]).float().to('cuda:0')
        data = {'z': pseudo_z.contiguous() + z_bias}
        with self.double_encoding:
            out = self.decode(data, initial_sampling, viz_att, viz_components)
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
