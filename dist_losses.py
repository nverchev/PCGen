# Distances and functions
import numpy as np
import torch
import torch.nn.functional as F
import geomloss
from abc import ABCMeta, abstractmethod
from sklearn import metrics
from pykeops.torch import LazyTensor


# tensor_dims = (batch, (samples,) vertices, coordinates )
def self_square_distance(t1):
    t2 = t1.transpose(-1, -2)
    dist = -2 * torch.matmul(t1, t2)
    sq = torch.sum(t1 ** 2, -1, keepdim=True)
    dist += sq
    dist += sq.transpose(-1, -2)
    return dist


# tensor_dims = (batch, (samples,) vertices, coordinates )
def square_distance(t1, t2):
    t2 = t2.transpose(-1, -2)
    dist = -2 * torch.matmul(t1, t2)
    dist += torch.sum(t1 ** 2, -1, keepdim=True)
    dist += torch.sum(t2 ** 2, -2, keepdim=True)
    return dist


# Chamfer Distance
def chamfer(dist):
    return torch.min(dist, axis=-1)[0].sum(-1).mean() \
           + torch.min(dist, axis=-2)[0].sum(-1).mean()


# Nll reconstruction
def nll(inputs, recons, pairwise_dist):
    n = inputs.size()[1]
    m = recons.size()[1]
    # 0.1192794 precomputed var of trainval dataset
    sigma2 = 0.1192794
    sigma6 = sigma2 ** 3
    pairwise_dist /= - 2 * sigma2
    lse = torch.logsumexp(pairwise_dist, axis=2)
    normalize = 0.5 * np.log(sigma6 * (2 * np.pi) ** 3) + np.log(m)
    return -lse.sum(1).mean() + n * (normalize)


# MMD with Gaussian kernel
def mmd_gaussian(t2, dist, bandwith=0.01):
    m = t2.size()[2]
    dist_t2 = self_square_distance(t2) / (2 * bandwith)
    pairwise_dist = dist / (2 * bandwith)
    mmd = -2 * (torch.exp(-pairwise_dist)).mean([1, 2])
    mmd += (torch.exp(-dist_t2).sum([1, 2]) - m) / (m * (m - 1))
    return mmd.mean()


def kld_loss(q_mu, q_logvar, freebits=2):
    KLD_matrix = -1 - q_logvar + q_mu ** 2 + q_logvar.exp()
    KLD_free_bits = F.softplus(KLD_matrix - 2 * freebits) + 2 * freebits
    KLD = 0.5 * KLD_matrix.mean(0).sum()
    KLD_free = 0.5 * KLD_free_bits.mean(0).sum()
    return KLD, KLD_free


class AbstractVAELoss(metaclass=ABCMeta):
    losses = ['Criterion', 'KLD']
    c_rec = 1
    c_reg = 0.01

    def __call__(self, outputs, inputs, targets):
        recons = outputs['recon']
        if len(recons.size()) == 4:
            n_samples = recons.size()[1]
            inputs = inputs.unsqueeze(1).expand(-1, n_samples, -1, -1)
        KLD, KLD_free = kld_loss(outputs['mu'], outputs['log_var'])
        recon_loss_dict = self.get_recon_loss(inputs, recons)
        recon_loss = recon_loss_dict[self.losses[2]]
        reg_lss = self.regularization(outputs)
        criterion = self.c_rec * recon_loss + KLD_free + self.c_reg * reg_lss
        if torch.isnan(criterion):
            print(outputs)
            raise
        return {
            'Criterion': criterion,
            'KLD': KLD,
            'reg': reg_lss,
            **recon_loss_dict
        }

    @abstractmethod
    def get_recon_loss(self, inputs, recons):
        pass

    def regularization(self, outputs):
        if 'trans' not in outputs.keys():
            loss = 0
        else:
            trans = outputs["trans"]
            trans_dim = trans.size()[-1]
            device = trans.device
            eye = torch.eye(trans_dim, device=device)
            diff = torch.bmm(trans, trans.transpose(2, 1)) - eye
            loss = torch.norm(diff, keepdim=True).mean(dim=(1, 2))
        return loss


class VAELossChamfer(AbstractVAELoss):
    losses = AbstractVAELoss.losses + ['Chamfer']
    c_rec = 100

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        return {'Chamfer': chamfer(pairwise_dist)}


# Negative Log Likelihood Distance
class VAELossNLL(AbstractVAELoss):
    losses = AbstractVAELoss.losses + ['NLL', 'Chamfer']

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        chamfer_loss = chamfer(pairwise_dist.detach())
        return {'NLL': nll(inputs, recons, pairwise_dist),
                'Chamfer': chamfer_loss}


class VAELossMMD(AbstractVAELoss):
    losses = AbstractVAELoss.losses + ['MMD', 'Chamfer']
    c_rec = 1
    mmd_gaussian = geomloss.SamplesLoss(loss="gaussian",
                                        blur=.05, diameter=2,
                                        backend="tensorized")

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        chamfer_loss = chamfer(pairwise_dist.detach())
        mmd = self.mmd_gaussian(inputs, recons)
        return {'MMD': mmd.mean(),
                'Chamfer': chamfer_loss}


class VAELossSinkhorn(AbstractVAELoss):
    losses = AbstractVAELoss.losses + ['Sinkhorn', 'Chamfer']
    c_rec = 70000
    sinkhorn = geomloss.SamplesLoss(loss="sinkhorn", p=2,
                                    blur=.03, diameter=2,
                                    scaling=.3, backend="tensorized")

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        chamfer_loss = chamfer(pairwise_dist.detach())
        # need to divide into batches
        inputs_list = inputs.chunk(4, 0)
        recon_list = recons.chunk(4, 0)
        sk_loss = 0
        for inp, rec in zip(inputs_list, recon_list):
            sk_loss += self.sinkhorn(inp, rec).mean()
        return {'Sinkhorn': sk_loss,
                'Chamfer': chamfer_loss}


def get_loss(recon_loss):
    recon_loss_dict = {
        "Chamfer": VAELossChamfer,
        "NLL": VAELossNLL,
        "MMD": VAELossMMD,
        'Sinkhorn': VAELossSinkhorn,
    }
    return recon_loss_dict[recon_loss]()

