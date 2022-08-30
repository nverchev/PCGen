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
    sigma2 = 0.0001
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
    losses = ['Criterion', 'KLD']
    c_rec = 1

    def __init__(self, c_kld):
        self.c_kld = c_kld

    def __call__(self, outputs, inputs, targets):
        recons = outputs['recon']
        if len(recons.size()) == 4:
            n_samples = recons.size()[1]
            inputs = inputs.unsqueeze(1).expand(-1, n_samples, -1, -1)
        KLD, KLD_free = kld_loss(outputs['mu'], outputs['log_var'])
        recon_loss_dict = self.get_recon_loss(inputs, recons)
        recon_loss = recon_loss_dict['recon']
        criterion = self.c_rec * recon_loss + self.c_kld * KLD_free
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
    losses = AbstractVAELoss.losses + ['Chamfer', 'Chamfer_Augmented']
    c_rec = 1

    def __init__(self, c_kld):
        super().__init__(c_kld)

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        squared, augmented = chamfer(inputs, recons, pairwise_dist)
        return {'recon': augmented,
                'Chamfer': squared,
                'Chamfer_Augmented': augmented}


class VAELossChamferAugmented(AbstractVAELoss):
    c_rec = 200

    def __init__(self, c_kld):
        super().__init__(c_kld)

    def get_recon_loss(self, inputs, recons):
        loss_list = super().get_recon_loss(inputs, recons)
        return loss_list.update({'recon': loss_list['Chamfer_Augmented']})


class VAELossChamferSmooth(VAELossChamferAugmented):
    losses = VAELossChamferAugmented.losses + ['Chamfer_Smooth']

    def __init__(self, c_kld):
        super().__init__(c_kld)

    def get_recon_loss(self, inputs, recons):
        pairwise_dist = square_distance(inputs, recons)
        squared, augmented = chamfer(inputs, recons, pairwise_dist)
        smooth = chamfer_smooth(inputs, recons, pairwise_dist)
        return {'recon': augmented,
                'Chamfer': squared,
                'Chamfer_norm': augmented,
                'Chamfer_Smooth': smooth}


class VAELossSinkhorn(VAELossChamferAugmented):
    losses = AbstractVAELoss.losses + ['Chamfer', 'Chamfer_Augmented', 'Sinkhorn']
    c_rec = 70000
    def __init__(self, c_kld):
        super().__init__(c_kld)
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=.03, diameter=2, scaling=.3, backend='tensorized')

    def get_recon_loss(self, inputs, recons):
        sk_loss = self.sinkhorn(inputs, recons).mean()
        return super().get_recon_loss(inputs, recons).update({'recon': sk_loss, 'Sinkhorn':sk_loss})

def get_vae_loss(recon_loss):
    recon_loss_dict = {
        'Chamfer': VAELossChamfer,
        'Chamfer_Augmented': VAELossChamferAugmented,
        'Chamfer_Smooth': VAELossChamferSmooth,
        'Sinkhorn': VAELossSinkhorn,
    }
    return recon_loss_dict[recon_loss]
