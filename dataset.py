import numpy as np
import torch
import os
import zipfile
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, Subset
import requests
import h5py
import glob2

try:
    from requests.packages.urllib3.exceptions import InsecureRequestWarning

    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
except:
    pass


def normalize(cloud):
    cloud -= cloud.mean(axis=0)
    cloud /= np.max(np.sqrt(np.sum(cloud ** 2, axis=1)))
    return cloud.astype(np.float32)


def random_rotation(cloud):
    theta = torch.pi * 2 * torch.rand(1)
    s = torch.sin(theta)
    rotation_matrix = torch.eye(2) * torch.cos(theta)
    rotation_matrix[0, 1] = -s
    rotation_matrix[1, 0] = s
    cloud[:, [0, 2]] = cloud[:, [0, 2]].mm(rotation_matrix)
    return cloud


def random_scale_translate(cloud):
    scale = torch.rand(1, 3) * 5 / 6 + 2 / 3
    translate = torch.rand(1, 3) * 0.4 - 0.2
    cloud *= scale
    cloud += translate
    return cloud


def jitter(cloud, sigma=0.01, clip=0.02):
    jitter = torch.randn(cloud.shape) * sigma
    cloud += torch.clamp(jitter, min=-clip, max=clip)
    return cloud


class BaseDataset(Dataset):

    def __init__(self, split, data_dir, n_points, rotation, noise):
        self.data_name = None
        self.split = split
        self.data_dir = data_dir
        self.n_points = n_points
        self.rotation = rotation
        self.noise = noise

    def load(self, split):
        pcs = []
        labels = []
        for h5_name in glob2.glob(os.path.join(self.data_dir, self.data_name, f'*{split}*.h5')):
            with h5py.File(h5_name, 'r+') as f:
                # Dataset is already normalized
                pc = f['data'][:].astype('float32')
                pc = pc[:self.n_points]
                label = f['label'][:].astype('int64')
            pcs.append(pc)
            labels.append(label)
        pcs = np.concatenate(pcs, axis=0)
        labels = np.concatenate(labels, axis=0)
        return pcs, labels

    def __len__(self):
        return self.pcd.shape[0]

    def __getitem__(self, index):
        cloud, label = self.pcd[index], self.labels[index]
        if self.rotation:
            random_rotation(cloud)
        if self.noise:
            cloud = random_scale_translate(cloud)
            cloud = jitter(cloud)
        return cloud, label


class Modelnet40Dataset(BaseDataset):

    def __init__(self, split, data_dir, n_points=2048, rotation=False, noise=False):
        super().__init__(split, data_dir, n_points, rotation, noise)
        self.data_name = 'modelnet40_hdf5_2048'
        self.pcd, self.labels = self.load(split)
        self.pcd = torch.from_numpy(self.pcd)
        self.labels = self.labels.ravel()
        assert len(self) == len(self.labels)

    def load(self, split):
        # validation is inside the train split
        if split == "trainval":
            split = "train"
        return super().load(split)


class ShapeNetDataset(BaseDataset):

    def __init__(self, split, data_dir, n_points=2048, rotation=False, noise=False):
        super().__init__(split, data_dir, n_points, rotation, noise)
        self.data_name = 'shapenetcorev2_hdf5_2048'
        self.pcd, self.labels = self.load(split)
        self.pcd = torch.from_numpy(self.pcd)
        self.labels = self.labels.ravel()
        assert len(self) == len(self.labels)

    def load(self, split):
        if split == "trainval":
            pcd1, labels1 = super().load(split="train")
            pcd2, labels2 = super().load(split="val")
            return np.concatenate([pcd1, pcd2], axis=0), np.concatenate([labels1, labels2], axis=0),
        else:
            return super().load(split=split)


def get_dataset(experiment, dataset, batch_size, val_every=6, dir_path="./", n_points=2048, noise=False):
    data_dir = os.path.join(dir_path, 'dataset')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if dataset == "modelnet40":
        zip_path = os.path.join(data_dir, 'modelnet40_hdf5_2048.zip')
        url = "https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097/?dl=1"
        PCDataset = Modelnet40Dataset
    else:
        assert dataset == "shapenet", "wrong dataset name"
        zip_path = os.path.join(data_dir, 'shapenetcorev2_hdf5_2048.zip')
        url = 'https://cloud.tsinghua.edu.cn/f/06a3c383dc474179b97d/?dl=1'
        PCDataset = ShapeNetDataset
    if not os.path.exists(zip_path):
        r = requests.get(url, verify=False)
        open(zip_path, 'wb').write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    pin_memory = torch.cuda.is_available()
    if experiment[:5] == 'final':
        train_dataset = PCDataset(data_dir=data_dir, split="trainval", n_points=n_points, rotation=True, noise=noise)
        test_dataset = PCDataset(data_dir=data_dir, split="test", n_points=n_points)
        train_loader = torch.utils.data.DataLoader(train_dataset, drop_last=True,
                                                   batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

        val_loader = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, pin_memory=pin_memory)

    else:
        train_dataset = PCDataset(data_dir=data_dir, split="train", n_points=n_points)

        if dataset == "modelnet40":
            num_train = len(train_dataset)
            train_idx = list(range(num_train))  # next line removes indices from train_idx
            # pop backwards to keep correct indexing
            val_idx = [train_idx.pop(i) for i in train_idx[::-val_every]]
            val_dataset = Subset(train_dataset, val_idx)
            train_dataset = Subset(train_dataset, train_idx)
        else:
            val_dataset = PCDataset(data_dir=data_dir, split="val", n_points=n_points)

        train_loader = torch.utils.data.DataLoader(train_dataset, drop_last=True,
                                                   batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

        test_loader = None

    return train_loader, val_loader, test_loader

# Old preproc

#
#
# import numpy as np
# import torch
# import os
# import zipfile
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.data import Dataset, Subset
# import requests
#
# try:
#     from requests.packages.urllib3.exceptions import InsecureRequestWarning
#
#     requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
# except:
#     pass
#
#
# class PCDDataset(Dataset):
#     def __init__(self, data):
#         self.pcd = [torch.tensor(cloud) for cloud in data["pcd"]]
#         self.labels = data["labels"]
#         assert len(self) == len(self.labels)
#
#     def __len__(self):
#         return len(self.pcd)
#
#     def __getitem__(self, index):
#         return self.pcd[index], self.labels[index]
#
#
# def preprocess(path, n_points):
#     cloud = np.loadtxt(path, delimiter=',', max_rows=n_points, usecols=(0, 1, 2))
#     cloud -= cloud.mean(axis=0)
#     cloud /= np.max(np.sqrt(np.sum(cloud ** 2, axis=1)))
#     return cloud.astype(np.float32)
#
#
# def get_dataset(experiment, batch_size, val_every=6, dir_path="./", download=False,
#                 minioClient=None, n_points=2048):
#     final = experiment[:5] == 'final'
#     if download == "from_zip":
#         zip_path = os.path.join(dir_path, 'modelnet40.zip')
#         if not os.path.exists(zip_path):
#             url = 'https://cloud.tsinghua.edu.cn/f/06a3c383dc474179b97d/?dl=1'
#             "https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097 /?dl=1"
#             r = requests.get(url, verify=False)
#             open(zip_path, 'wb').write(r.content)
#         data_path = os.path.join(dir_path, 'modelnet40')
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(dir_path)
#         for split in ["train", "test"]:
#             names = []
#             shapes = []
#             pcd = []
#             with open(os.path.join(data_path, f'modelnet40_{split}.txt'), 'r') as f:
#                 for line in f.readlines():
#                     line = line.strip()
#                     names.append(line)
#                     shape = "_".join(line.split("_")[:-1])
#                     shapes.append(shape)
#                     path = os.path.join(data_path, shape, line) + ".txt"
#                     pcd.append(preprocess(path, n_points))
#             sorted_shapes = sorted(set(shapes))  # both dataset have the same shapes
#             label_map = {name: label for label, name in enumerate(sorted_shapes)}
#             labels = [label_map[shape] for shape in shapes]
#             ds_name = f'{split}_dataset.npz'
#             np.savez(ds_name, pcd=pcd, labels=labels, names=names)
#     elif download == "from_minio":
#         assert minioClient is not None, "Please provide minio client"
#         minioClient.fget_object('pcdvae', 'train_dataset.npz', 'train_dataset.npz')
#         minioClient.fget_object('pcdvae', 'test_dataset.npz', 'test_dataset.npz')
#     else:
#         assert os.path.exists(dir_path + 'train_dataset.npz'), "Dataset not found"
#
#     pin_memory = torch.cuda.is_available()
#     train_data = np.load(dir_path + 'train_dataset.npz', allow_pickle=True)
#     test_data = np.load(dir_path + 'test_dataset.npz', allow_pickle=True)
#     train_dataset = PCDDataset(data=train_data)
#     test_dataset = PCDDataset(data=test_data)
#     num_train = len(train_dataset)
#     train_idx = list(range(num_train))  # next line removes indices from train_idx
#     # pop backwards to keep correct indexing
#     val_idx = [train_idx.pop(i) for i in train_idx[::-val_every]]
#     initial_train_dataset = Subset(train_dataset, train_idx)
#     val_dataset = Subset(train_dataset, val_idx)
#     if final:
#         train_loader = torch.utils.data.DataLoader(train_dataset, drop_last=True,
#                                                    batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
#
#         val_loader = None
#
#     else:
#         train_loader = torch.utils.data.DataLoader(initial_train_dataset, drop_last=True,
#                                                    batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
#
#         val_loader = torch.utils.data.DataLoader(val_dataset,
#                                                  batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
#
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
#                                               shuffle=False, pin_memory=pin_memory)
#
#     return train_loader, val_loader, test_loader
