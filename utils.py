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
    t2 = t1.transpose(-1, -2)
    square_component = torch.sum(t1 ** 2, -1, keepdim=True)
    dist = -2 * torch.matmul(t1, t2)
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
    idx = d_ij.topk(k=k, dim=-1)[1]
    return idx


def get_graph_features(x, k=20):
    x = x.transpose(2, 1).contiguous()
    batch, n_points, n_feat = x.size()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx = idx.view(batch, k * n_points, 1).expand(-1, -1,  n_feat)
    feature = torch.gather(x, 1, idx).view(batch, n_points, k, n_feat)
    x_feat = x.unsqueeze(2).expand(-1, -1, k, -1)
    feature = torch.cat((feature - x_feat, x_feat), dim=3).permute(0, 3, 1, 2).contiguous()
    # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
    return feature  # (batch_size, 2*num_dims, num_points, k)

