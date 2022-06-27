# @title Libraries
import torch
import torch.nn as nn
from encoder import get_encoder
from decoder import get_decoder


class VAE(nn.Module):
    settings = {}

    def __init__(self, encoder_name, decoder_name):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.encode = get_encoder(encoder_name)()
        self.decode = get_decoder(decoder_name)()

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
        if isinstance(x, list):
            data['trans'] = x[1]
            x = x[0]
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
