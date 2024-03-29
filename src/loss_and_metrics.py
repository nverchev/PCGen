import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import geomloss
from src.neighbour_op import pykeops_square_distance, cpu_square_distance
# from emd import emdModule
from structural_losses import match_cost


# Chamfer Distance
def pykeops_chamfer(t1, t2):
    # The following code is currently not supported for backprop
    # return (dist.min(axis = 2) + dist.min(axis = 1)).sum(axis=(1, 2)
    # We use the retrieved index on torch
    dist = pykeops_square_distance(t1, t2)
    idx1 = dist.argmin(axis=1).expand(-1, -1, t1.shape[2])
    m1 = t1.gather(1, idx1)
    squared1 = ((t2 - m1) ** 2).sum(axis=(1, 2))
    # augmented1 = torch.sqrt(((t2 - m1) ** 2).sum(-1)).mean(1)
    idx2 = dist.argmin(axis=2).expand(-1, -1, t1.shape[2])
    m2 = t2.gather(1, idx2)
    squared2 = ((t1 - m2) ** 2).sum(axis=(1, 2))
    # augmented2 = torch.sqrt(((t1 - m2) ** 2).sum(-1)).mean(1)
    # forward + reverse
    squared = squared1 + squared2
    # augmented = torch.maximum(augmented1, augmented2)
    return squared  # , augmented


# Works with distance in torch
def cpu_chamfer(t1, t2):
    dist = cpu_square_distance(t1, t2)
    return torch.min(dist, dim=-1)[0].sum(1) + torch.min(dist, dim=-2)[0].sum(1)


def gaussian_ll(x, mu, log_var):
    return -0.5 * (log_var + torch.pow(x - mu, 2) / torch.exp(log_var))


def kld_loss(mu, log_var, z, pseudo_mu, pseudo_log_var, d_mu=(), d_log_var=(), prior_log_var=(), **_):
    posterior_ll = gaussian_ll(z, mu, log_var).sum(1)  # sum dimensions
    k = pseudo_mu.shape[0]
    b = mu.shape[0]
    z = z.unsqueeze(1).expand(-1, k, -1)  # expand to each component
    pseudo_mu = pseudo_mu.unsqueeze(0).expand(b, -1, -1)  # expand to each sample
    pseudo_log_var = pseudo_log_var.unsqueeze(0).expand(b, -1, -1)  # expand to each sample
    prior_ll = torch.logsumexp(gaussian_ll(z, pseudo_mu, pseudo_log_var).sum(2), dim=1)
    kld = posterior_ll - prior_ll + np.log(k)
    for d_mu, d_log_var, p_log_var in zip(d_mu, d_log_var, prior_log_var):
        # d_mu = q_mu - p_mu
        # d_logvar = q_logvar - p_logvar
        kld_matrix = -1 - d_log_var + (d_mu ** 2) / p_log_var.exp() + d_log_var.exp()
        kld += 0.5 * kld_matrix.sum(1)
    return kld


# as a class because of the sinkhorn loss init
class ReconLoss:
    c_rec = 1

    def __init__(self, recon_loss_name, device):
        self.recon_loss_name = recon_loss_name
        self.device = device
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=.001,
                                             diameter=2, scaling=.9, backend='online')
        # self.emd_dist = emdModule()
        self.chamfer = pykeops_chamfer if device.type == 'cuda' else cpu_chamfer

    def __call__(self, inputs, recon):
        squared = self.chamfer(inputs, recon)
        dict_recon = {'Chamfer': squared}
        if self.recon_loss_name == 'Chamfer' or self.device.type == 'cpu':
            recon_loss = squared
        elif self.recon_loss_name == 'ChamferEMD':
            # emd = torch.sqrt(self.emd_dist(inputs, recon, 0.005, 50)[0]).sum(1)
            emd = match_cost(inputs.contiguous(), recon.contiguous())
            recon_loss = emd + squared
            dict_recon['EMD'] = emd
        else:
            assert self.recon_loss_name == 'Sinkhorn', f'Loss {self.recon_loss_name} not known'
            sk_loss = self.sinkhorn(inputs, recon)
            recon_loss = sk_loss
            dict_recon['Sinkhorn'] = sk_loss

        dict_recon['recon'] = recon_loss
        return dict_recon


class AELoss(nn.Module):
    def __init__(self, recon_loss_name, device, **not_used):
        super().__init__()
        self.recon_loss = ReconLoss(recon_loss_name, device)

    def forward(self, outputs, inputs, targets):
        ref_cloud = inputs[-2]  # get input shape (resampled depending on the dataset)
        recon = outputs['recon']
        recon_loss_dict = self.recon_loss(ref_cloud, recon)
        return {
            'Criterion': recon_loss_dict.pop('recon'),
            **recon_loss_dict
        }


# VAE is only for the second encoding
# class VAELoss(AELoss):
#     def __init__(self, get_recon_loss, get_reg_loss, c_reg):
#         super().__init__(get_recon_loss)
#         self.get_reg_loss = get_reg_loss
#         self.c_kld = c_reg
#
#     def forward(self, outputs, inputs, targets):
#         recon_loss_dict = super().forward(outputs, inputs, targets)
#         kld = kld_loss(*[outputs[key] for key in ['mu', 'log_var']])
#         reg = self.c_kld * kld
#         criterion = recon_loss_dict.pop('Criterion') + reg
#         return {'Criterion': criterion,
#                 **recon_loss_dict,
#                 'reg': reg,
#                 'KLD': kld,
#                 }


class VQVAELoss(AELoss):
    def __init__(self, recon_loss_name, c_commitment, c_embedding, vq_ema_update, device, **_):
        super().__init__(recon_loss_name, device)
        self.c_commitment = c_commitment
        self.c_embedding = c_embedding
        self.vq_ema_update = vq_ema_update

    def forward(self, outputs, inputs, targets):
        recon_loss_dict = super().forward(outputs, inputs, targets)
        if self.vq_ema_update or self.c_embedding == self.c_commitment:
            commit_loss = F.mse_loss(outputs['cw_q'], outputs['cw_e'])
            reg = self.c_commitment * commit_loss
        else:
            commit_loss = F.mse_loss(outputs['cw_q'], outputs['cw_e'].detach())
            embed_loss = F.mse_loss(outputs['cw_q'].detach(), outputs['cw_e'])
            reg = self.c_commitment * commit_loss + self.c_embedding * embed_loss
        criterion = recon_loss_dict.pop('Criterion') + reg
        return {'Criterion': criterion,
                **recon_loss_dict,
                'Embed Loss': commit_loss * outputs['cw_q'].shape[0]  # embed_loss and commit_loss have same value
                }


class CWEncoderLoss(nn.Module):
    def __init__(self, c_kld, **_):
        super().__init__()
        self.c_kld = c_kld

    def forward(self, outputs, *_):
        kld = kld_loss(**outputs)
        one_hot_idx = outputs['one_hot_idx'].clone()
        sqrt_dist = torch.sqrt(outputs['cw_dist'])
        cw_neg_dist = -sqrt_dist + sqrt_dist.min(2, keepdim=True)[0]
        nll = -(cw_neg_dist.log_softmax(dim=2) * one_hot_idx).sum((1, 2))
        criterion = nll + self.c_kld * kld
        one_hot_predictions = F.one_hot(outputs['cw_dist'].argmin(2), num_classes=one_hot_idx.shape[2])
        accuracy = (one_hot_idx * one_hot_predictions).sum(2).mean(1)
        return {
            'Criterion': criterion,
            'KLD': kld,
            'NLL': nll,
            'Accuracy': accuracy
        }


def get_ae_loss(model_head, **args):
    return (AELoss if model_head in ('AE', 'Oracle') else VQVAELoss)(**args)


class AllMetrics:
    def __init__(self, de_normalize):
        self.de_normalize = de_normalize
        # self.emd_dist = emdModule()

    def __call__(self, outputs, inputs):
        scale = inputs[0]
        ref_cloud = inputs[-2]  # get input shape (resampled depending on the dataset)
        recon = outputs['recon']
        if self.de_normalize:
            scale = scale.view(-1, 1, 1).expand_as(recon)
            recon *= scale
            ref_cloud *= scale
        dict_metrics = self.batched_pairwise_similarity(ref_cloud, recon)
        if self.de_normalize:
            dict_metrics = {'Denorm ' + k: v for k, v in dict_metrics.items()}
        return dict_metrics

    @staticmethod
    def batched_pairwise_similarity(clouds1, clouds2):
        if clouds1.device.type == 'cpu':
            squared = cpu_chamfer(clouds1, clouds2)
            warnings.warn('Emd only supports cuda tensors', category=RuntimeWarning)
            emd = torch.zeros(1, 1)
        else:
            squared = pykeops_chamfer(clouds1, clouds2)
            emd = match_cost(clouds1.contiguous(), clouds2.contiguous())
            # emd = torch.sqrt(self.emd_dist(clouds1, clouds2, 0.005, 50)[0]).mean(1)
        if clouds1.shape == clouds2.shape:
            squared /= clouds1.shape[1]  # Chamfer is normalised by ref and recon number of points when equal
            emd /= clouds1.shape[1]  # Chamfer is normalised by ref and recon number of points when equal
        dict_recon_metrics = {
            'Chamfer': squared,
            'EMD': emd
        }
        return dict_recon_metrics
