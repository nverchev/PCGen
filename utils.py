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

def knn(x, k):
    d_ij = square_distance(x, x)
    idx = d_ij.argKmin(k, dim=2)
    return idx


def get_graph_features(x, k=20):

    x = x.transpose(2, 1)
    batch, n_points, n_feat = x.size()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx = idx.view(batch, k * n_points, 1).expand(-1, -1,  n_feat)
    feature = torch.gather(x, 1, idx).view(batch, n_points, k, n_feat)
    x_feat = x.unsqueeze(2).expand(-1, -1, k, -1)
    # important contiguous after permute
    feature = torch.cat((feature - x_feat, x_feat), dim=3).permute(0, 3, 1, 2).contiguous()
    # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
    return feature  # (batch_size, 2*num_dims, num_points, k)



def knn1(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx



def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn1(x, k=k)  # (batch_size, num_points, k)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2)  # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature
