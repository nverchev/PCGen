# Distances and functions
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import geomloss
from src.neighbour_op import square_distance
from emd import emdModule
from structural_losses import match_cost

# Chamfer Distance
def chamfer(t1, t2, dist):
    # The following code is currently not supported for backprop
    # return dist.min(axis = 2).mean(0).sum()\
    #      + dist.min(axis = 1).mean(0).sum()
    # We use the retrieved index on torch
    idx1 = dist.argmin(axis=1).expand(-1, -1, t1.shape[2])
    m1 = t1.gather(1, idx1)
    squared1 = ((t2 - m1) ** 2).sum(axis=(1, 2))
    augmented1 = torch.sqrt(((t2 - m1) ** 2).sum(-1)).mean(1)
    idx2 = dist.argmin(axis=2).expand(-1, -1, t1.shape[2])
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


#  def chamfer_smooth(inputs, recon, pairwise_dist):
#     n = inputs.size()[1]
#     m = recon.size()[1]
#     # variance of the components (model assumption)
#     sigma2 = 0.001
#     idx2 = pairwise_dist.argmin(axis=2).expand(-1, -1, 3)
#     m2 = recon.gather(1, idx2)
#     sigma2 = ((inputs - m2) ** 2).mean().detach().item()
#     pairwise_dist /= - 2 * sigma2
#     lse1 = pairwise_dist.logsumexp(axis=2)
#     normalize1 = 1.5 * np.log(sigma2 * 2 * np.pi) + np.log(m)
#     loss1 = -lse1.sum(1) + n * normalize1
#     lse2 = pairwise_dist.logsumexp(axis=1)
#     normalize2 = 1.5 * np.log(sigma2 * 2 * np.pi) + np.log(n)
#     loss2 = -lse2.sum(2) + n * normalize2
#     return (loss1 + loss2).sum(1)

def gaussian_ll(x, mean, log_var):
    return -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))


def kld_loss(mu, log_var, z, pseudo_mu, pseudo_log_var, d_mu=(), d_log_var=(), prior_log_var=(), **not_used):
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

    def __init__(self, recon_loss='Chamfer'):
        self.recon_loss = recon_loss
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=.001,
                                             diameter=2, scaling=.9, backend='online')
        self.emd_dist = emdModule()

    def __call__(self, inputs, recon):
        pairwise_dist = square_distance(inputs, recon)
        squared, augmented = chamfer(inputs, recon, pairwise_dist)
        dict_recon = {'Chamfer': squared.sum(0), 'Chamfer Augmented': augmented.sum(0)}
        if self.recon_loss == 'Chamfer':
            recon = squared.mean(0)  # Sum over points, mean over samples
        elif self.recon_loss == 'ChamferEMD':
            #emd = torch.sqrt(self.emd_dist(inputs, recon, 0.005, 50)[0]).sum(1)  # mean over samples
            emd = match_cost(inputs.contiguous(), recon.contiguous())
            recon = emd.mean(0) + squared.mean(0)  # Sum over points, mean over samples
            dict_recon['EMD'] = emd.sum(0)
        else:
            assert self.recon_loss == 'Sinkhorn', f'Loss {self.recon_loss} not known'
            sk_loss = self.sinkhorn(inputs, recon)
            recon = sk_loss.mean(0)
            dict_recon['Sinkhorn'] = sk_loss.sum(0)

        dict_recon['recon'] = recon
        return dict_recon


class AELoss(nn.Module):
    def __init__(self, recon_loss, **not_used):
        super().__init__()
        self.recon_loss = recon_loss

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
#         reg = self.c_kld * kld.mean(0)
#         criterion = recon_loss_dict.pop('Criterion') + reg
#         return {'Criterion': criterion,
#                 **recon_loss_dict,
#                 'reg': reg,
#                 'KLD': kld.sum(0),
#                 }


class VQVAELoss(AELoss):
    def __init__(self, recon_loss, c_commitment, c_embedding, vq_ema_update, **not_used):
        super().__init__(recon_loss)
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
    def __init__(self, c_kld, **not_used):
        super().__init__()
        self.c_kld = c_kld

    def forward(self, outputs, inputs, targets):
        kld = kld_loss(**outputs)
        one_hot_idx = outputs['one_hot_idx'].clone() #inputs[1]
        sqrt_dist = torch.sqrt(outputs['cw_dist'])
        cw_neg_dist = -sqrt_dist + sqrt_dist.min(2, keepdim=True)[0]
        nll = -(cw_neg_dist.log_softmax(dim=2) * one_hot_idx).sum((1, 2))
        criterion = nll.mean(0) + self.c_kld * kld.mean(0)
        one_hot_predictions = F.one_hot(outputs['cw_dist'].argmin(2), num_classes=one_hot_idx.shape[2])
        accuracy = (one_hot_idx * one_hot_predictions).sum(2).mean(1)
        return {
            'Criterion': criterion,
            'KLD': kld.sum(0),
            'NLL': nll.sum(0),
            'Accuracy': accuracy.sum(0)
        }


# Currently not used, replace VQVAE loo in get_ae_loss to use it
class DoubleEncodingLoss(VQVAELoss):
    def __init__(self, recon_loss, c_commitment, c_embedding, c_kld, vq_ema_update, **not_used):
        super().__init__(recon_loss, c_commitment=c_commitment, c_embedding=c_embedding, vq_ema_update=vq_ema_update)
        self.cw_loss = CWEncoderLoss(c_kld)

    def forward(self, outputs, inputs, targets):
        dict_loss = super().forward(outputs, inputs, targets)
        second_dict_loss = self.cw_loss(outputs, [None, outputs['one_hot_idx']], targets)
        criterion = dict_loss.pop('Criterion') + second_dict_loss.pop('Criterion')
        return {'Criterion': criterion,
                **dict_loss,
                **second_dict_loss}


def get_ae_loss(model_head, recon_loss, **other_args):
    recon_loss = ReconLoss(recon_loss)
    return (AELoss if model_head in ('AE', 'Oracle') else VQVAELoss)(recon_loss, **other_args)
    #return (AELoss if model_head in ('AE', 'Oracle') else DoubleEncodingLoss)(recon_loss, **other_args)


class AllMetrics:
    def __init__(self, de_normalize):
        self.de_normalize = de_normalize
        # self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=.01, diameter=2, scaling=.9, backend='online')
        # self.emd_dist = emdModule()

    def __call__(self, outputs, inputs):
        scale = inputs[0]
        ref_cloud = inputs[-2]  # get input shape (resampled depending on the dataset)
        recon = outputs['recon']
        if self.de_normalize:
            scale = scale.view(-1, 1, 1).expand_as(recon)
            recon *= scale
            ref_cloud *= scale
        return self.batched_pairwise_similarity(ref_cloud, recon)

    @staticmethod
    def batched_pairwise_similarity(clouds1, clouds2):
        pairwise_dist = square_distance(clouds1, clouds2)
        squared, augmented = chamfer(clouds1, clouds2, pairwise_dist)
        emd = match_cost(clouds1.contiguous(), clouds2.contiguous())
        # emd = torch.sqrt(self.emd_dist(clouds1, clouds2, 0.005, 50)[0]).mean(1)
        if clouds1.shape == clouds2.shape:
            squared /= clouds1.shape[1]  # Chamfer is normalised by ref and recon number of points when equal
            emd /= clouds1.shape[1]  # Chamfer is normalised by ref and recon number of points when equal
        dict_recon_metrics = {
            'Chamfer': squared.sum(0),
            'Chamfer Augmented': augmented.sum(0),
            'EMD': emd.sum(0)
        }
        return dict_recon_metrics
