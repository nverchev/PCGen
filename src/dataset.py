import numpy as np
import torch
import os
import zipfile
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, Subset
import requests
import h5py
import glob2
from sklearn.neighbors import KDTree

try:
    from requests.packages.urllib3.exceptions import InsecureRequestWarning

    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
except:
    pass


def download_zip(dir_path, zip_name, url):
    data_dir = os.path.join(dir_path, '../dataset')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    zip_path = os.path.join(data_dir, zip_name)
    if not os.path.exists(zip_path):
        r = requests.get(url, verify=False)
        open(zip_path, 'wb').write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    return zip_path[:-4]


def index_k_neighbours(pcs, k):
    indices_list = []
    for pc in pcs:
        kdtree = KDTree(pc)
        indices = kdtree.query(pc, k, return_distance=False)
        indices_list.append(indices.reshape(-1, k))
    return np.stack(indices_list, axis=0)


def load_h5(wild_path, num_points, k):
    pcd = []
    indices = []
    labels = []
    for h5_name in glob2.glob(wild_path):
        with h5py.File(h5_name, 'r+') as f:
            print('Load: ', h5_name)
            # Dataset is already normalized
            pcs = f['data'][:].astype('float32')
            pcs = pcs[:, :num_points, :]
            label = f['label'][:].astype('int64')
            index_k = f'index_{k}'
            if index_k in f.keys():
                index = f[index_k][:].astype(np.short)
            else:
                index = index_k_neighbours(pcs, k).astype(np.short)
                f.create_dataset(index_k, data=index)

        pcd.append(pcs)
        indices.append(index)
        labels.append(label)
    pcd = np.concatenate(pcd, axis=0)
    indices = np.concatenate(indices, axis=0)
    labels = np.concatenate(labels, axis=0)
    return pcd, indices, labels.ravel()


def normalize(cloud):
    cloud -= cloud.mean(axis=0)
    cloud /= np.max(np.sqrt(np.sum(cloud ** 2, axis=1)))
    return cloud.astype(np.float32)


def random_rotation(cloud):
    theta = 2 * torch.pi * torch.rand(1)
    s = torch.sin(theta)
    rotation_matrix = torch.eye(2) * torch.cos(theta)
    rotation_matrix[0, 1] = -s
    rotation_matrix[1, 0] = s
    new_cloud = cloud.clone()
    new_cloud[:, [0, 2]] = cloud[:, [0, 2]].mm(rotation_matrix)
    return new_cloud


def random_scale_translate(cloud):
    scale = torch.rand(1, 3) * 5 / 6 + 2 / 3
    translate = torch.rand(1, 3) * 0.4 - 0.2
    new_cloud = cloud.clone()
    new_cloud *= scale
    new_cloud += translate
    return new_cloud


def jitter(cloud, sigma=0.01, clip=0.02):
    jitter = torch.randn(cloud.shape) * sigma
    new_cloud = cloud.clone()
    new_cloud += torch.clamp(jitter, min=-clip, max=clip)
    return new_cloud

class CoinDataset:
    def __init__(self, half_thickness=0.1, n_points=2048, k=20, **other_settings):
        self.half_thickness = half_thickness
        self.width = 2 * self.half_thickness
        self.radius = np.sqrt(1 - self.half_thickness ** 2)
        self.n_points = n_points
        self.coin = self.create_coin().astype(np.single)
        self.neighbours = torch.from_numpy(index_k_neighbours([self.coin], k)).long()

    def create_coin(self):
        sides = round(self.n_points * 2 * self.radius / (2 * self.radius + 4 * self.half_thickness))
        thetas = 2 * np.pi * np.random.rand(sides)
        rs = np.sqrt(np.random.rand(sides))
        exps = rs * np.exp(thetas * 1j)
        x1, y1 = exps.real, exps.imag
        z1 = np.random.choice([self.half_thickness, -self.half_thickness], size=sides, replace=True)
        thetas = 2 * np.pi * np.random.rand(self.n_points - sides)
        exps = np.exp(thetas * 1j)
        x2, y2 = exps.real, exps.imag
        z2 = (2 * np.random.rand(self.n_points - sides) - 1) * self.half_thickness
        x, y, z = np.hstack([x1, x2]), np.hstack([y1, y2]), np.hstack([z1, z2])
        return np.stack([x, y, z], axis=1)

    def create_hole(self, coin, pt):
        global hole1_index, hole1, hole
        hole_index = np.logical_and(abs(coin[:, 0] - pt[0]) < self.width, abs(coin[:, 1] - pt[1]) < self.width)
        hole = coin[hole_index]
        hole1_index = hole[:, 2] == self.half_thickness
        hole2_index = np.logical_not(hole1_index)
        hole1 = hole[hole1_index]
        hole2 = hole[hole2_index]
        pt = np.tile(pt.reshape(1, 3), [hole1.shape[0], 1])

        close_side1 = np.where(np.expand_dims(hole1[:, 0] < pt[:, 0], axis=1),
                               np.stack(
                                   [pt[:, 0] - self.width, hole1[:, 1], pt[:, 0] - hole1[:, 0] - self.half_thickness],
                                   axis=1),
                               np.stack(
                                   [pt[:, 0] + self.width, hole1[:, 1], hole1[:, 0] - pt[:, 0] - self.half_thickness],
                                   axis=1))

        pt = np.tile(pt[0].reshape(1, 3), [hole2.shape[0], 1])
        close_side2 = np.where(np.expand_dims(hole2[:, 1] < pt[:, 1], axis=1),
                               np.stack(
                                   [hole2[:, 0], pt[:, 1] - self.width, pt[:, 1] - hole2[:, 1] - self.half_thickness],
                                   axis=1),
                               np.stack(
                                   [hole2[:, 0], self.width + pt[:, 1], hole2[:, 1] - pt[:, 1] - self.half_thickness],
                                   axis=1))
        hole[hole1_index] = close_side1
        hole[hole2_index] = close_side2
        coin[hole_index] = hole
        return coin

    def sample_point(self):
        radius = self.radius - np.sqrt(2) * self.width
        theta = 2 * np.pi * np.random.rand(1)
        expo = radius * np.exp(theta * 1j)
        x, y = expo.real, expo.imag
        z = np.random.choice([self.half_thickness, -self.half_thickness], 1)
        return np.hstack([x, y, z])

    def __len__(self):
        return 1024

    def __getitem__(self, index):
        n_holes = np.random.randint(low=1, high=4)
        coin = self.coin.copy()
        pts = []
        for _ in range(n_holes):
            while True:
                pt = self.sample_point()
                if any(abs(past_pt[0] - pt[0]) < self.width or abs(past_pt[1] - pt[1]) < self.width for past_pt in pts):
                    continue
                pts.append(pt)
                self.create_hole(coin, pt)
                break
        plate = torch.from_numpy(coin)
        return [plate, self.neighbours], n_holes


class PCDataset(Dataset):
    def __init__(self, pcd, indices, labels, rotation, translation):
        self.pcd = pcd
        self.indices = indices
        self.labels = labels
        self.rotation = rotation
        self.translation_and_scale = translation

    def __len__(self):
        return self.pcd.shape[0]

    def __getitem__(self, index):
        cloud, index, label = self.pcd[index], self.indices[index], self.labels[index]
        cloud = torch.from_numpy(cloud)
        index = torch.from_numpy(index).long()
        if self.rotation:
            random_rotation(cloud)
        if self.translation_and_scale:
            cloud = random_scale_translate(cloud)
        return [cloud, index], label


class Modelnet40Dataset:

    def __init__(self, dir_path, k, num_points, **augmentation_settings):
        self.dir_path = dir_path
        self.data_name = 'modelnet40_hdf5_2048'
        self.augmentation_settings = augmentation_settings
        data_path = self.download()
        files_path = lambda x: os.path.join(data_path, f'*{x}*.h5')
        self.pcd, self.indices, self.labels = {}, {}, {}
        for split in ['train', 'test']:
            self.pcd[split], self.indices[split], self.labels[split] = load_h5(files_path(split), num_points, k)

    def split(self, split):
        if split == 'trainval':
            assert 'val' not in self.pcd.keys(), "train dataset has already been split"
            split = 'train'
        elif split in ['train', 'val'] and 'val' not in self.pcd.keys():
            self.trainval_to_train_and_val()
        return PCDataset(pcd=self.pcd[split], indices=self.indices[split],
                         labels=self.labels[split], **self.augmentation_settings)

    def download(self):
        url = 'https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097/?dl=1'
        return download_zip(dir_path=self.dir_path, zip_name=self.data_name + '.zip', url=url)

    def trainval_to_train_and_val(self, val_every=6):
        train_idx = list(range(self.pcd['train'].shape[0]))
        val_idx = [train_idx.pop(i) for i in train_idx[::-val_every]]
        # partition train into train and val
        for new_split, new_split_idx in [['val', val_idx], ['train', train_idx]]:
            self.pcd[new_split] = self.pcd['train'][new_split_idx]
            self.indices[new_split] = self.indices['train'][new_split_idx]
            self.labels[new_split] = self.labels['train'][new_split_idx]


class ShapeNetDataset:

    def __init__(self, dir_path, k, num_points, **augmentation_settings):
        self.dir_path = dir_path
        self.data_name = 'shapenetcorev2_hdf5_2048'
        self.augmentation_settings = augmentation_settings
        data_path = self.download()
        files_path = lambda x: os.path.join(data_path, f'*{x}*.h5')
        self.pcd, self.indices, self.labels = {}, {}, {}
        for split in ['train', 'val', 'test']:
            self.pcd[split], self.indices[split], self.labels[split] = load_h5(files_path(split), num_points, k)
        self.pcd['trainval'] = np.concatenate([self.pcd['train'], self.pcd['val']], axis=0)
        self.indices['trainval'] = np.concatenate([self.indices['train'], self.indices['val']], axis=0)
        self.labels['trainval'] = np.concatenate([self.labels['train'], self.labels['val']], axis=0)

    def split(self, split):
        return PCDataset(pcd=self.pcd[split], indices=self.indices[split],
                         labels=self.labels[split], **self.augmentation_settings)

    def download(self):
        url = 'https://cloud.tsinghua.edu.cn/f/06a3c383dc474179b97d/?dl=1'
        return download_zip(dir_path=self.dir_path, zip_name=self.data_name + '.zip', url=url)


def get_dataset(dataset_name, batch_size, final, **dataset_settings):
    pin_memory = torch.cuda.is_available()
    if dataset_name == 'coin':
        dataset = CoinDataset(**dataset_settings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)
        return loader, loader, loader
    if dataset_name == 'modelnet40':
        dataset = Modelnet40Dataset(**dataset_settings)
    elif dataset_name == 'shapenet':
        dataset = ShapeNetDataset(**dataset_settings)
    else:
        print(dataset_name)
        raise ValueError()

    if final:
        train_dataset = dataset.split('trainval')
        train_loader = torch.utils.data.DataLoader(
            train_dataset, drop_last=True, batch_size=batch_size,
            shuffle=True, pin_memory=pin_memory)
        val_loader = None
    else:
        train_dataset = dataset.split('train')
        val_dataset = dataset.split('val')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, drop_last=True, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, drop_last=False, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    test_dataset = dataset.split('test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, drop_last=False,
        shuffle=False, pin_memory=pin_memory)
    del dataset
    return train_loader, val_loader, test_loader
