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
    idx = d_ij.topk(k=k, largest=False, dim=-1)[1]
    return idx


def get_graph_features(x, k=20):
    batch, n_feat, n_points = x.size()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx = idx.view(batch, 1, k * n_points).expand(-1,  n_feat, -1)
    feature = torch.gather(x, 2, idx).view(batch, n_feat, n_points, k)
    x = x.unsqueeze(3).expand(-1, -1, -1, k)
    feature = torch.cat([feature - x, x], dim=1).contiguous()
    # (batch_size, 2*num_dims, num_points, k)
    return feature

# def local_cov(x, k=16):
#     batch, n_feat, n_points = x.size()
#     idx = knn(x, k=k)  # (batch_size, num_points, k)
#     idx = idx.view(batch, 1, k * n_points).expand(-1, n_feat, -1)
#     feature = torch.gather(x, 2, idx).view(batch, n_feat, n_points, k)
#     feature -= feature.mean(-1, keepdim=True)
#     torch.matmul(feature.transpose(2, 3), feature).view(batch, n, -1).permute(0, 2, 1)
#     # (batch_size, num_points, k, 3)
#     x = torch.matmul(x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
#     # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
#     x = x.view(batch_size, num_points, 9).transpose(2, 1)  # (batch_size, 9, num_points)
#
#     x = torch.cat((feature, x), dim=1)  # (batch_size, 12, num_points)