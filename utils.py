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
    x_expanded = x.unsqueeze(1).expand(-1, n_points, -1, -1)
    idx = idx.unsqueeze(3).expand(-1, -1, -1, n_feat)
    feature = torch.gather(x_expanded, 2, idx)
    x_feat = x.unsqueeze(2).expand(-1, -1, k, -1)
    feature = torch.cat((feature - x_feat, x_feat), dim=3).permute(0, 3, 1, 2)
    # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature  # (batch_size, 2*num_dims, num_points, k)
