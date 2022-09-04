import torch
from pykeops.torch import LazyTensor


# tensor_dims = (batch, (samples,) vertices, coordinates )
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
#     return dist
#


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
    indices = d_ij.topk(k=k, largest=False, dim=-1)[1].contiguous()
    return indices

def get_neighbours(x, k, indices):
    batch, n_feat, n_points = x.size()
    if indices is not None:
        indices = indices
    else:
        indices = knn(x, k=k)  # (batch_size, num_points, k)
        indices = indices.view(batch, 1, k * n_points).expand(-1,  n_feat, -1)
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

