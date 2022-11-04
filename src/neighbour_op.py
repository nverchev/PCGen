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
#     # [batch, points, features]
#     t2 = t2.transpose(-1, -2)
#     dist = -2 * torch.matmul(t1, t2)
#     dist += torch.sum(t1 ** 2, -1, keepdim=True)
#     dist += torch.sum(t2 ** 2, -2, keepdim=True)
#     return dist
#
# def self_square_distance(t1):
#     t2 = t1.transpose(-1, -2)
#     square_component = torch.sum(t1 ** 2, -2, keepdim=True)
#     dist = -2 * torch.matmul(t2, t1)
#     dist += square_component
#     dist += square_component.transpose(-1, -2)
#     return dist
#
#
# def knn(x, k):
#     d_ij = self_square_distance(x)
#     return d_ij.topk(k=k, largest=False, dim=-1)

def knn(x, k):
    x = x.transpose(2, 1)
    d_ij = square_distance(x, x)
    indices = d_ij.argKmin(k, dim=2)
    return  indices


def get_neighbours(x, k, indices):
    batch, n_feat, n_points = x.size()
    if indices is not None:
        indices = indices
    else:
        indices = knn(x, k)  # (batch_size, num_points, k)
    indices = indices.contiguous().view(batch, 1, k * n_points).expand(-1, n_feat, -1)
    neighbours = torch.gather(x, 2, indices).view(batch, n_feat, n_points, k)
    return neighbours


def get_local_covariance(x, k=16, indices=None):
    neighbours = get_neighbours(x, k, indices)
    neighbours -= neighbours.mean(3, keepdim=True)
    covariances = torch.matmul(neighbours.transpose(1, 2), neighbours.permute(0, 2, 3, 1))
    x = torch.cat([x, covariances.flatten(start_dim=2).transpose(1, 2)], dim=1).contiguous()
    return x


def graph_max_pooling(x, k=16, indices=None):
    neighbours = get_neighbours(x, k, indices)
    max_pooling = torch.max(neighbours, dim=-1)[0]
    return max_pooling


def get_graph_features(x, k=20, indices=None):
    neighbours = get_neighbours(x, k, indices)
    x = x.unsqueeze(3).expand(-1, -1, -1, k)
    feature = torch.cat([neighbours - x, x], dim=1).contiguous()
    # (batch_size, 2 * num_dims, num_points, k)
    return feature


def graph_filtering(x, k=4):
    neighbours = get_neighbours(x, k=k, indices=None)
    neighbours = neighbours[..., 1:]  # closest neighbour is point itself
    dist = torch.sqrt(((x.unsqueeze(-1).expand(-1, -1, -1, k-1) - neighbours)**2).sum(1))
    sigma = dist.mean()
    weights = torch.softmax(-dist / sigma, dim=-1)
    weighted_neighbours = weights.unsqueeze(1).expand(-1, 3, -1, -1) * neighbours
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
