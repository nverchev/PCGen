# Distances and functions
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import geomloss
from src.neighbour_op import square_distance
from external.emd.emd_module import emdModule


# Chamfer Distance

def chamfer(t1, t2, dist):
    # The following code is currently not supported for backprop
    # return dist.min(axis = 2).mean(0).sum()\
    #      + dist.min(axis = 1).mean(0).sum()
    # We use the retrieved index on torch
    idx1 = dist.argmin(axis=1).expand(-1, -1, 3)
    m1 = t1.gather(1, idx1)
    squared1 = ((t2 - m1) ** 2).sum(axis=(1, 2))
    augmented1 = torch.sqrt(((t2 - m1) ** 2).sum(-1)).mean(1)
    idx2 = dist.argmin(axis=2).expand(-1, -1, 3)
    m2 = t2.gather(1, idx2)
    squared2 = ((t1 - m2) ** 2).sum(axis=(1, 2))
    augmented2 = torch.sqrt(((t1 - m2) ** 2).sum(-1)).mean(1)
    # forward + reverse
    squared = squared1 + squared2
    augmented = torch.maximum(augmented1, augmented2)
    return squared, augmented


# # Works with distance in torch
# def chamfer(t1, t2, dist):
#     return torch.min(dist, axis=-1)[0].mean() \
#            +  torch.min(dist, axis=-2)[0].mean()


# Nll recontruction
def chamfer_smooth(inputs, recon, pairwise_dist):
    n = inputs.size()[1]
    m = recon.size()[1]
    # variance of the components (model assumption)
    sigma2 = 0.001
    pairwise_dist /= - 2 * sigma2
    lse1 = pairwise_dist.logsumexp(axis=2)
    normalize1 = 1.5 * np.log(sigma2 * 2 * np.pi) + np.log(m)
    loss1 = -lse1.sum(1) + n * normalize1
    lse2 = pairwise_dist.logsumexp(axis=1)
    normalize2 = 1.5 * np.log(sigma2 * 2 * np.pi) + np.log(n)
    loss2 = -lse2.sum(2) + n * normalize2
    return (loss1 + loss2).sum(1)


def kld_loss(q_mu, q_log_var, d_mu=(), d_log_var=(), p_log_var=()):
    kld_matrix = -1 - q_log_var + q_mu ** 2 + q_log_var.exp()
    kld = 0.5 * kld_matrix.sum(1)
    for d_mu, d_log_var, p_log_var in zip(d_mu, d_log_var, p_log_var):
        # d_mu = q_mu - p_mu
        # d_logvar = q_logvar - p_logvar
        kld_matrix = -1 - d_log_var + (d_mu ** 2) / p_log_var.exp() + d_log_var.exp()
        kld += 0.5 * kld_matrix.sum(1)
    return kld


class VAELoss(nn.Module):
    def __init__(self, get_reg_loss, get_recon_loss, c_reg):
        super().__init__()
        self.get_reg_loss = get_reg_loss
        self.get_recon_loss = get_recon_loss
        self.c_reg = c_reg

    def forward(self, outputs, inputs, targets):
        ref_cloud = inputs[-2]  # get input shape (resampled depending on the dataset)
        recon = outputs['recon']
        # if recon.dim == 4:
        #     n_samples = recon.size()[1]
        #     inputs = inputs.unsqueeze(1).expand(-1, n_samples, -1, -1)
        reg_loss_dict = self.get_reg_loss(ref_cloud, outputs)
        reg_loss = reg_loss_dict.pop('reg')
        recon_loss_dict = self.get_recon_loss(ref_cloud, recon)
        recon_loss = recon_loss_dict.pop('recon')
        criterion = recon_loss + self.c_reg * reg_loss
        return {
            'Criterion': criterion,
            **reg_loss_dict,
            **recon_loss_dict
        }


class AELoss:
    def __call__(self, inputs, outputs):
        return {'reg': torch.tensor(0)}


class KLDVAELoss:
    c_kld = 1

    def __call__(self, inputs, outputs):
        kld = kld_loss(*[outputs[key] for key in ['mu', 'log_var']])
        return {'reg': self.c_kld * kld.mean(0),
                'KLD': kld.sum(0),
                }


class VQVAELoss:
    c_vq = .5

    def __call__(self, inputs, outputs):
        embed_loss = F.mse_loss(outputs['cw_q'], outputs['cw_e'])
        return {'reg': self.c_vq * embed_loss,
                'Embed Loss': embed_loss * outputs['cw_q'].shape[0],
                }


class ReconLoss:
    c_rec = 1

    def __init__(self, recon_loss='Chamfer'):
        self.recon_loss = recon_loss
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=.001,
                                             diameter=2, scaling=.9, backend='online')

    def __call__(self, inputs, recon):
        pairwise_dist = square_distance(inputs, recon)
        squared, augmented = chamfer(inputs, recon, pairwise_dist)
        dict_recon = {'Chamfer': squared.sum(0), 'Chamfer Augmented': augmented.sum(0)}
        if self.recon_loss == 'Chamfer':
            recon = squared.mean(0)
        elif self.recon_loss == 'ChamferA':
            recon = augmented.mean(0)
        elif self.recon_loss == 'ChamferS':
            smooth = chamfer_smooth(inputs, recon, pairwise_dist)
            recon = smooth.mean(0)
            dict_recon['Chamfer Smooth'] = smooth.sum(0)
        else:
            assert self.recon_loss == 'Sinkhorn', f'Loss {self.recon_loss} not known'
            sk_loss = self.sinkhorn(inputs, recon)
            recon = sk_loss.mean(0)
            dict_recon['Sinkhorn'] = sk_loss.sum(0)

        dict_recon['recon'] = recon
        return dict_recon


class CWEncoderLoss(nn.Module):
    def __init__(self, c_reg):
        super().__init__()
        self.c_reg = c_reg

    def forward(self, outputs, inputs, targets):
        kld = kld_loss(*[outputs[key] for key in ['mu', 'log_var', 'd_mu', 'd_log_var', 'prior_log_var']])
        cw_idx = inputs[1]
        cw_neg_dist = -outputs['cw_dist']
        nll = -(cw_neg_dist.log_softmax(dim=2) * cw_idx).sum((1, 2))
        criterion = nll.mean(0) + self.c_reg * kld.mean(0)
        return {
            'Criterion': criterion,
            'KLD': kld.sum(0),
            'NLL': nll.sum(0),
        }


def get_ae_loss(block_args):
    get_recon_loss = ReconLoss(block_args['recon_loss'])
    if block_args['ae'] in ('AE', 'Oracle'):
        get_reg_loss = AELoss()
    elif block_args['ae'] == 'VQVAE':
        get_reg_loss = VQVAELoss()
    else:
        get_reg_loss = KLDVAELoss()
    return VAELoss(get_reg_loss, get_recon_loss, block_args['c_reg'])


class AllMetrics:
    def __init__(self, denormalise):
        self.denormalise = denormalise
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=.01,
                                             diameter=2, scaling=.6, backend='online')
        self.emd = emdModule()

    def __call__(self, outputs, inputs):
        scale = inputs[0]
        ref_cloud = inputs[-2]  # get input shape (resampled depending on the dataset)
        recon = outputs['recon']
        if self.denormalise:
            scale = scale.view(-1, 1, 1).expand_as(recon)
            recon *= scale
            ref_cloud *= scale
        return self.batched_pairwise_similarity(ref_cloud, recon)

    def batched_pairwise_similarity(self, clouds1, clouds2):
        pairwise_dist = square_distance(clouds1, clouds2)
        squared, augmented = chamfer(clouds1, clouds2, pairwise_dist)
        if clouds1.shape == clouds2.shape:
            squared /= clouds1.shape[1]  # Chamfer is normalised by ref and recon number of points when equal
        dict_recon_metrics = {
            'Chamfer': squared.sum(0),
            'Chamfer Augmented': augmented.sum(0),
            'Chamfer Smooth': chamfer_smooth(clouds1, clouds2, pairwise_dist).sum(0),
            'Sinkhorn': self.sinkhorn(clouds1, clouds2).sum(0),
            # 'EMD': self.emd(clouds1, clouds2, 0.05, 3000)[0].mean(1).sum(0),
        }
        return dict_recon_metrics
