import numpy as np
import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor

def square_distance(t1, t2):
    t1 = LazyTensor(t1[:, :, None, :])
    t2 = LazyTensor(t2[:, None, :, :])
    dist = ((t1 - t2) ** 2).sum(-1)
    return dist


# def square_distance(t1, t2):
#     t2 = t2.transpose(-1, -2)
#     dist = -2 * torch.matmul(t1, t2)
#     dist += torch.sum(t1 ** 2, -1, keepdim=True)
#     dist += torch.sum(t2 ** 2, -2, keepdim=True)
#     return dist.unsqueeze(3)


def self_square_distance(t1):
    # dims order is different than square distance: [batch, features, points]
    t2 = t1.transpose(-1, -2)
    square_component = torch.sum(t1 ** 2, 1, keepdim=True)
    dist = -2 * torch.matmul(t2, t1)
    dist += square_component
    dist += square_component.transpose(-1, -2)
    return dist


# Not very efficient (not sure why)
# def knn(x, k):
#     d_ij = square_distance(x, x)
#     idx = d_ij.argKmin(k, dim=2)
#     return idx


def knn(x, k):
    d_ij = self_square_distance(x)
    indices = d_ij.topk(k=k, largest=False, dim=-1)
    return indices


def get_neighbours(x, k, indices):
    batch, n_feat, n_points = x.size()
    if indices is not None:
        indices = indices
        dist = None
    else:
        dist, indices = knn(x, k=k)  # (batch_size, num_points, k)
    indices = indices.contiguous().view(batch, 1, k * n_points).expand(-1, n_feat, -1)
    neighbours = torch.gather(x, 2, indices).view(batch, n_feat, n_points, k)
    return dist, neighbours


def get_local_covariance(x, k=16, indices=None):
    neighbours = get_neighbours(x, k, indices)[1]
    neighbours -= neighbours.mean(3, keepdim=True)
    covariances = torch.matmul(neighbours.transpose(1, 2), neighbours.permute(0, 2, 3, 1))
    x = torch.cat([x, covariances.flatten(start_dim=2).transpose(1, 2)], dim=1).contiguous()
    return x


def graph_max_pooling(x, k=16, indices=None):
    neighbours = get_neighbours(x, k, indices)[1]
    max_pooling = torch.max(neighbours, dim=-1)[0]
    return max_pooling


def get_graph_features(x, k=20, indices=None):
    neighbours = get_neighbours(x, k, indices)[1]
    x = x.unsqueeze(3).expand(-1, -1, -1, k)
    feature = torch.cat([neighbours - x, x], dim=1).contiguous()
    # (batch_size, 2 * num_dims, num_points, k)
    return feature

def graph_filtering(x):
    dist, neighbours = get_neighbours(x, k=4, indices=None)
    dist1 = dist[..., 1:]  # dist[:, :,  0] == 0
    neighbours1 = neighbours[..., 1:]
    sigma = torch.sqrt(torch.clamp(dist1, min=0.00001).detach()).mean(-1, keepdims=True)
    weights = torch.softmax(-dist1 / sigma, dim=-1)
    weighted_neighbours = weights.unsqueeze(1).expand(-1, 3, -1, -1) * neighbours1
    x = 1.5 * x - 0.5 * weighted_neighbours.sum(-1)
    return x

# def graph_filtering(x):
#     minus_dist = -self_square_distance(x)
#     sigma2 = np.sqrt(0.001)
#     exclude_self = torch.arange(x.size(2), device=x.device).view(1, -1, 1).expand(x.size(0), -1, -1)
#     minus_dist.scatter_(2, exclude_self, -torch.inf)
#     weights = torch.softmax(minus_dist / sigma2, dim=1)
#     weighted_neighbours = torch.bmm(x, weights)
#     x = 1.5 * x - 0.5 * weighted_neighbours
#     return x
