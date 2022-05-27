import numpy as np
import torch
import os
import zipfile
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, Subset


class PCDDataset(Dataset):
    def __init__(self, data):
        self.pcd = [torch.tensor(cloud) for cloud in data["pcd"]]
        self.labels = data["labels"]
        assert len(self) == len(self.labels)

    def __len__(self):
        return len(self.pcd)

    def __getitem__(self, index):
        return self.pcd[index], self.labels[index]


def preprocess(path, n_points):
    cloud = np.loadtxt(path, delimiter=',', max_rows=n_points, usecols=(0, 1, 2))
    cloud -= cloud.mean(axis=0)
    cloud /= np.max(np.sqrt(np.sum(cloud ** 2, axis=1)))
    return cloud.astype(np.float32)


def get_dataset(batch_size, val_split=1/6, dirpath="./", minioClient=None, dataset_from_zip=False, n_points=2048):
    if dataset_from_zip:
        zip_path = os.path.join(dirpath, 'modelnet40_normal_resampled.zip')
        data_path = os.path.join(dirpath, 'modelnet40_normal_resampled')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dirpath)
        for split in ["train", "test"]:
            names = []
            shapes = []
            pcd = []
            with open(os.path.join(data_path, f'modelnet40_{split}.txt'), 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    names.append(line)
                    shape = "_".join(line.split("_")[:-1])
                    shapes.append(shape)
                    path = os.path.join(data_path, shape, line) + ".txt"
                    pcd.append(preprocess(path, n_points))
            sorted_shapes = sorted(set(shapes))  # both dataset have the same shapes
            label_map = {name: label for label, name in enumerate(sorted_shapes)}
            labels = [label_map[shape] for shape in shapes]
            ds_name = f'{split}_dataset.npz'
            np.savez(ds_name, pcd=pcd, labels=labels, names=names)
    elif minioClient:
        minioClient.fget_object('pcdvae', 'train_dataset.npz', 'train_dataset.npz')
        minioClient.fget_object('pcdvae', 'test_dataset.npz', 'test_dataset.npz')
    else:
        assert os.path.exists('train_dataset.npz'), "Dataset not found"

    pin_memory = torch.cuda.is_available()
    train_data = np.load(dirpath + 'train_dataset.npz', allow_pickle=True)
    test_data = np.load(dirpath + 'test_dataset.npz', allow_pickle=True)
    train_dataset = PCDDataset(data=train_data)
    test_dataset = PCDDataset(data=test_data)
    num_train = len(train_dataset)
    train_idx = list(range(num_train))  # next line removes indices from train_idx
    val_split_jump = round(1/val_split)
    val_idx = [train_idx.pop(i) for i in train_idx[::val_split_jump]]
    initial_train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(train_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(initial_train_dataset, drop_last=True,
                                               batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    train_val_loader = torch.utils.data.DataLoader(train_dataset, drop_last=True,
                                                   batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, pin_memory=pin_memory)
    return train_loader, val_loader, train_val_loader, test_loader
