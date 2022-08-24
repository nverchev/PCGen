# Distances and functions
import numpy as np
import torch
import torch.nn.functional as F
import geomloss
from abc import ABCMeta, abstractmethod
from utils import square_distance


# Chamfer Distance

def chamfer(t1, t2, dist):
    # The following code is currently not supported for backprop
    # return dist.min(axis = 2).mean(0).sum()\
    #      + dist.min(axis = 1).mean(0).sum()
    # We use the retrieved index on torch
    idx1 = dist.argmin(axis=1).expand(-1, -1, 3)
    m1 = t1.gather(1, idx1)
    squared1 = ((t2 - m1) ** 2).mean(0).sum()
    norm1 = (torch.sqrt((t2 - m1) ** 2).sum(-1)).mean()
    idx2 = dist.argmin(axis=2).expand(-1, -1, 3)
    m2 = t2.gather(1, idx2)
    squared2 = ((t1 - m2) ** 2).mean(0).sum()
    norm2 = (torch.sqrt((t1 - m2) ** 2).sum(-1)).mean()
    # forward + reverse
    squared = squared1 + squared2
    norm = norm1 + norm2
    return squared, norm


# # Works with distance in torch
# def chamfer(t1, t2, dist):
#     return torch.min(dist, axis=-1)[0].mean() \
#            +  torch.min(dist, axis=-2)[0].mean()


# Nll reconstruction
def nll(inputs, recons, pairwise_dist):
    n = inputs.size()[1]
    m = recons.size()[1]
    # variance of the components (model assumption)
    sigma2 = 0.0001
    pairwise_dist /= - 2 * sigma2
    lse = pairwise_dist.logsumexp(axis=2)
    normalize = 1.5 * np.log(sigma2 * 2 * np.pi) + np.log(m)
    return -lse.sum(1).mean() + n * normalize


# def nll(inputs, recons, pairwise_dist):
#     n = inputs.size()[1]
#     m = recons.size()[1]
#     # 0.1192794 precomputed var of trainval dataset
#     sigma2 = 0.1192794
#     sigma6 = sigma2 ** 3
#     pairwise_dist /= - 2 * sigma2
#     lse = torch.logsumexp(pairwise_dist, axis=2)
#     normalize = 0.5 * np.log(sigma6 * (2 * np.pi) ** 3) + np.log(m)
#     return -lse.sum(1).mean() + n * normalize

def kld_loss(q_mu, q_logvar, freebits=2):
    KLD_matrix = -1 - q_logvar + q_mu ** 2 + q_logvar.exp()
    KLD_free_bits = F.softplus(KLD_matrix - 2 * freebits) + 2 * freebits
    KLD = 0.5 * KLD_matrix.mean(0).sum()
    KLD_free = 0.5 * KLD_free_bits.mean(0).sum()
    return KLD, KLD_free


class CalLoss:
    eps = 0.2
    losses = ['Criterion', "Calib Loss"]

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, outputs, inputs, targets):
        logits = outputs['y']
        one_hot = torch.zeros_like(logits).scatter(1, targets.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (self.num_classes - 1)
        log_prb = F.log_softmax(logits, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return {'Criterion': loss,
                'Calib Loss': loss}


class AbstractVAELoss(metaclass=ABCMeta):
    losses = ['Criterion', 'KLD']
    c_rec = 1
    c_KLD = 0.001

    def __call__(self, outputs, inputs, targets):
        recons = outputs['recon']
        if len(recons.size()) == 4:
            n_samples = recons.size()[1]
            inputs = inputs.unsqueeze(1).expand(-1, n_samples, -1, -1)
        KLD, KLD_free = kld_loss(outputs['mu'], outputs['log_var'])
        recon_loss_dict = self.get_recon_loss(inputs, recons)
        recon_loss = recon_loss_dict['recon']
        criterion = self.c_rec * recon_loss + self.c_KLD * KLD_free
        if torch.isnan(criterion):
            print(outputs)
            raise
        return {
            'Criterion': criterion,
            'KLD': KLD,
            **recon_loss_dict
        }

    @abstractmethod
    def get_recon_loss(self, inputs, recons):
        pass


class VAELossChamfer(AbstractVAELoss):
    losses = AbstractVAELoss.losses + ['Chamfer', 'Chamfer_norm']
    c_rec = 1

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        squared, norm = chamfer(inputs, recons, pairwise_dist)
        return {'recon': squared,
                'Chamfer': squared,
                'Chamfer_norm': norm}


# Negative Log Likelihood Distance
class VAELossNLL(AbstractVAELoss):
    losses = AbstractVAELoss.losses + ['NLL', 'Chamfer', 'Chamfer_norm']

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        squared, norm = chamfer(inputs, recons, pairwise_dist)
        recon = nll(inputs, recons, pairwise_dist)
        return {'recon':recon,
                'NLL': recon,
                'Chamfer': squared,
                'Chamfer_norm': norm}


class VAELossSinkhorn(AbstractVAELoss):
    losses = AbstractVAELoss.losses + ['Sinkhorn', 'Chamfer', 'Chamfer_norm']
    c_rec = 70000
    sinkhorn = geomloss.SamplesLoss(loss="sinkhorn", p=2,
                                    blur=.03, diameter=2,
                                    scaling=.3, backend="tensorized")

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        squared, norm = chamfer(inputs, recons, pairwise_dist)
        # need to divide into batches
        inputs_list = inputs.chunk(4, 0)
        recon_list = recons.chunk(4, 0)
        sk_loss = 0
        for inp, rec in zip(inputs_list, recon_list):
            sk_loss += self.sinkhorn(inp, rec).mean()
        return {'recon': sk_loss,
                'Sinkhorn': sk_loss,
                'Chamfer': squared,
                'Chamfer_norm': norm}

def get_classification_loss(loss):
    classification_loss_dict = {
        "cal": CalLoss,
    }
    return classification_loss_dict[loss]


def get_vae_loss(recon_loss):
    recon_loss_dict = {
        "Chamfer": VAELossChamfer,
        "NLL": VAELossNLL,
        'Sinkhorn': VAELossSinkhorn,
    }
    return recon_loss_dict[recon_loss]()
