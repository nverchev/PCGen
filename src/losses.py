# Distances and functions
import numpy as np
import torch
import torch.nn.functional as F
import geomloss
from abc import ABCMeta, abstractmethod
from src.utils import square_distance


# Chamfer Distance

def chamfer(t1, t2, dist):
    # The following code is currently not supported for backprop
    # return dist.min(axis = 2).mean(0).sum()\
    #      + dist.min(axis = 1).mean(0).sum()
    # We use the retrieved index on torch
    idx1 = dist.argmin(axis=1).expand(-1, -1, 3)
    m1 = t1.gather(1, idx1)
    squared1 = ((t2 - m1) ** 2).mean(0).sum()
    augmented1 = torch.sqrt(((t2 - m1) ** 2).sum(-1)).mean()
    idx2 = dist.argmin(axis=2).expand(-1, -1, 3)
    m2 = t2.gather(1, idx2)
    squared2 = ((t1 - m2) ** 2).mean(0).sum()
    augmented2 = torch.sqrt(((t1 - m2) ** 2).sum(-1)).mean()
    # forward + reverse
    squared = squared1 + squared2
    augmented = max(augmented1, augmented2)
    return squared, augmented


# # Works with distance in torch
# def chamfer(t1, t2, dist):
#     return torch.min(dist, axis=-1)[0].mean() \
#            +  torch.min(dist, axis=-2)[0].mean()


# Nll reconstruction
def chamfer_smooth(inputs, recons, pairwise_dist):
    n = inputs.size()[1]
    m = recons.size()[1]
    # variance of the components (model assumption)
    sigma2 = 0.001
    pairwise_dist /= - 2 * sigma2
    lse1 = pairwise_dist.logsumexp(axis=2)
    normalize1 = 1.5 * np.log(sigma2 * 2 * np.pi) + np.log(m)
    loss1 = -lse1.sum(1).mean() + n * normalize1
    lse2 = pairwise_dist.logsumexp(axis=1)
    normalize2 = 1.5 * np.log(sigma2 * 2 * np.pi) + np.log(n)
    loss2 = -lse2.sum(2).mean() + n * normalize2
    return loss1 + loss2


def kld_loss(q_mu, q_logvar, freebits=2):
    KLD_matrix = -1 - q_logvar + q_mu ** 2 + q_logvar.exp()
    KLD_free_bits = F.softplus(KLD_matrix - 2 * freebits) + 2 * freebits
    KLD = 0.5 * KLD_matrix.mean(0).sum()
    KLD_free = 0.5 * KLD_free_bits.mean(0).sum()
    return KLD, KLD_free


class AbstractVAELoss(metaclass=ABCMeta):
    def __init__(self, c_reg):
        self.c_reg = c_reg

    def __call__(self, outputs, inputs, targets):
        recons = outputs['recon']
        if len(recons.size()) == 4:
            n_samples = recons.size()[1]
            inputs = inputs.unsqueeze(1).expand(-1, n_samples, -1, -1)
        reg_loss_dict = self.get_reg_loss(self, inputs, outputs)
        reg_loss = reg_loss_dict.pop('reg')
        recon_loss_dict = self.get_recon_loss(inputs, recons)
        recon_loss = recon_loss_dict.pop('recon')
        criterion = self.c_rec * recon_loss + self.c_reg * reg_loss
        if torch.isnan(criterion):
            print(outputs)
            raise
        return {
            'Criterion': criterion,
            **reg_loss_dict,
            **recon_loss_dict
        }
    @abstractmethod
    def get_reg_loss(self, inputs, recons):
        pass
    @abstractmethod
    def get_recon_loss(self, inputs, recons):
        pass


class KLDVAELoss(metaclass=ABCMeta):
    def get_reg_loss(self, inputs, outputs):
        KLD, KLD_free = kld_loss(outputs['mu'], outputs['log_var'])
        return {'reg': KLD_free,
                'KLD': KLD
                }

class VAELossChamfer():
    c_rec = 1

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        squared, augmented = chamfer(inputs, recons, pairwise_dist)
        return {'recon': squared,
                'Chamfer': squared,
                'Chamfer_Augmented': augmented}


class VAELossChamferAugmented(VAELossChamfer):
    c_rec = 1000

    def get_recon_loss(self, inputs, recons):
        loss_dict = super().get_recon_loss(inputs, recons)
        loss_dict.update({'recon': loss_dict['Chamfer_Augmented']})
        return loss_dict


class VAELossChamferSmooth(VAELossChamfer):
    c_rec = 0.1

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        squared, augmented = chamfer(inputs, recons, pairwise_dist)
        smooth = chamfer_smooth(inputs, recons, pairwise_dist)
        return {'recon': smooth,
                'Chamfer': squared,
                'Chamfer_Augmented': augmented,
                'Chamfer_Smooth': smooth}


class VAELossSinkhorn(VAELossChamferAugmented):
    c_rec = 1000
    sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=.03, diameter=2, scaling=.3, backend='tensorized')

    def get_recon_loss(self, inputs, recons):
        sk_loss = self.sinkhorn(inputs, recons).mean()
        loss_dict = super().get_recon_loss(inputs, recons)
        loss_dict.update({'recon': sk_loss, 'Sinkhorn': sk_loss})
        return loss_dict


def get_vae_loss(recon_loss, vq=False):
    recon_loss_dict = {
        'Chamfer': VAELossChamfer,
        'Chamfer_A': VAELossChamferAugmented,
        'Chamfer_S': VAELossChamferSmooth,
        'Sinkhorn': VAELossSinkhorn,
    }
    class FinalLoss(recon_loss_dict[recon_loss], KLDVAELoss):
        pass 
    return FinalLoss
