import numpy as np
import torch
import os
from torch.utils.data import Dataset
from src.utils import download_zip, index_k_neighbours, load_h5_modelnet, load_h5_dfaust
import glob2
import openmesh


def normalize(cloud):
    cloud -= cloud.mean(axis=0)
    cloud /= np.max(np.sqrt(np.sum(cloud ** 2, axis=1)))
    return cloud.astype(np.float32)


def random_rotation(*clouds):
    theta = 2 * torch.pi * torch.rand(1)
    s = torch.sin(theta)
    rotation_matrix = torch.eye(2) * torch.cos(theta)
    rotation_matrix[0, 1] = -s
    rotation_matrix[1, 0] = s
    new_clouds = []
    for cloud in clouds:
        new_cloud = cloud.clone()
        new_cloud[:, [0, 2]] = cloud[:, [0, 2]].mm(rotation_matrix)
        new_clouds.append(new_cloud)
    return new_clouds


def random_scale_translate(*clouds):
    scale = torch.rand(1, 3) * 5 / 6 + 2 / 3
    translate = torch.rand(1, 3) * 0.4 - 0.2
    new_clouds = []
    for cloud in clouds:
        new_cloud = cloud.clone()
        new_cloud *= scale
        new_cloud += translate
        new_clouds.append(new_cloud)
    return new_clouds


def jitter(cloud, sigma=0.01, clip=0.02):
    jitter = torch.randn(cloud.shape) * sigma
    new_cloud = cloud.clone()
    new_cloud += torch.clamp(jitter, min=-clip, max=clip)
    return new_cloud


class CWDataset(Dataset):
    def __init__(self, cw_q, cw_e, labels, filter_class=None):
        self.cw_q = torch.stack(cw_q)
        self.cw_e = torch.stack(cw_e)
        self.labels = torch.stack(labels)
        if filter_class is not None:
            idx = (self.labels == filter_class).squeeze()
            self.cw_q = self.cw_q[idx]
            self.cw_e = self.cw_e[idx]

    def __len__(self):
        return self.cw_q.shape[0]

    def __getitem__(self, index):
        return [self.cw_q[index], self.cw_e[index]], self.labels[index]


class PiercedCoinsDataset(Dataset):
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
        return 4096

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
            cloud, = random_rotation(cloud)
        if self.translation_and_scale:
            cloud, = random_scale_translate(cloud)
        return [cloud, index], label


class Modelnet40Dataset:

    def __init__(self, data_dir, k, num_points, **augmentation_settings):
        self.data_dir = data_dir
        self.modelnet_path = os.path.join(data_dir, 'modelnet40_hdf5_2048')
        self.augmentation_settings = augmentation_settings
        self.download()
        files_path = lambda x: os.path.join(self.modelnet_path, f'*{x}*.h5')
        self.pcd, self.indices, self.labels = {}, {}, {}
        for split in ['train', 'test']:
            self.pcd[split], self.indices[split], self.labels[split] = \
                load_h5_modelnet(files_path(split), num_points, k)

    def split(self, split):
        if split == 'trainval':
            assert 'val' not in self.pcd.keys(), 'train dataset has already been split'
            split = 'train'
        elif split in ['train', 'val'] and 'val' not in self.pcd.keys():
            self.trainval_to_train_and_val()
        return PCDataset(pcd=self.pcd[split], indices=self.indices[split],
                         labels=self.labels[split], **self.augmentation_settings)

    def download(self):
        url = 'https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097/?dl=1'
        return download_zip(data_dir=self.data_dir, zip_path=self.modelnet_path + '.zip', url=url)

    def trainval_to_train_and_val(self, val_every=6):
        train_idx = list(range(self.pcd['train'].shape[0]))
        val_idx = [train_idx.pop(i) for i in train_idx[::-val_every]]
        # partition train into train and val
        for new_split, new_split_idx in [['val', val_idx], ['train', train_idx]]:
            self.pcd[new_split] = self.pcd['train'][new_split_idx]
            self.indices[new_split] = self.indices['train'][new_split_idx]
            self.labels[new_split] = self.labels['train'][new_split_idx]


class ShapeNetOldDataset:

    def __init__(self, data_dir, k, num_points, **augmentation_settings):
        self.data_dir = data_dir
        self.shapenet_path = os.path.join(self.data_dir, 'shapenetcorev2_hdf5_2048')
        self.augmentation_settings = augmentation_settings
        self.download()
        files_path = lambda x: os.path.join(self.shapenet_path, f'{x}*.h5')
        print(files_path('test'))
        self.pcd, self.indices, self.labels = {}, {}, {}
        for split in ['train', 'val', 'test']:
            self.pcd[split], self.indices[split], self.labels[split] = \
                load_h5_modelnet(files_path(split), num_points, k)
        self.pcd['trainval'] = np.concatenate([self.pcd['train'], self.pcd['val']], axis=0)
        self.indices['trainval'] = np.concatenate([self.indices['train'], self.indices['val']], axis=0)
        self.labels['trainval'] = np.concatenate([self.labels['train'], self.labels['val']], axis=0)

    def split(self, split):
        return PCDataset(pcd=self.pcd[split], indices=self.indices[split],
                         labels=self.labels[split], **self.augmentation_settings)

    def download(self):
        url = 'https://cloud.tsinghua.edu.cn/f/06a3c383dc474179b97d/?dl=1'
        return download_zip(data_dir=self.data_dir, zip_path=self.shapenet_path + '.zip', url=url)


class PCDatasetResampled(Dataset):
    def __init__(self, paths, num_points, labels, rotation, translation):
        self.paths = paths
        self.rotation = rotation
        self.translation_and_scale = translation
        self.num_points = num_points
        self.label_index = list(labels.values())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        cloud = np.load(path)
        index_pool = np.arange(cloud.shape[0])
        sampling_recon = np.random.choice(index_pool, size=self.num_points, replace=False)
        sampling_target = np.random.choice(index_pool, size=self.num_points, replace=False)
        cloud_recon = torch.from_numpy(normalize(cloud[sampling_recon]))
        cloud_input = torch.from_numpy(normalize(cloud[sampling_target]))
        if self.rotation:
            cloud_recon, cloud_input = random_rotation(cloud_recon, cloud_input)
        if self.translation_and_scale:
            cloud_recon, cloud_input = random_scale_translate(cloud_recon, sampling_target)
        label = os.path.split(os.path.split(path)[0])[1]
        label = self.label_index.index(label)
        return [cloud_recon, cloud_input, 0], label


class ShapeNetDataset:
    labels = {
        '02958343': 'car',
        '03001627': 'chair',
        '03211117': 'monitor',
        '03636649': 'lamp',
        '03691459': 'speaker',
        '04090263': 'firearm',
        '04256520': 'couch',
        '04379243': 'table',
        '04401088': 'cellphone',
        '04530566': 'watercraft',
        '02691156': 'plane',
        '02828884': 'bench',
        '02933112': 'cabinet'
    }

    def __init__(self, data_dir, k, num_points, **augmentation_settings):
        self.data_dir = data_dir
        self.shapenet_path = os.path.join(self.data_dir, 'shapenet')
        if not os.path.exists(self.shapenet_path):
            os.mkdir(self.shapenet_path)
            self.to_numpy()
        self.val_ratio = 0.2
        self.test_ratio = 0.2
        self.augmentation_settings = augmentation_settings
        self.num_points = num_points
        folders = glob2.glob(os.path.join(self.shapenet_path, '*'))
        self.paths = {}
        for folder in folders:
            files = glob2.glob(os.path.join(folder, '*'))
            first_split = int(len(files) * (1 - self.val_ratio - self.test_ratio))
            second_split = int(len(files) * (1 - self.test_ratio))
            self.paths.setdefault('train', []).extend(files[:first_split])
            self.paths.setdefault('trainval', []).extend(files[:second_split])
            self.paths.setdefault('val', []).extend(files[first_split:second_split])
            self.paths.setdefault('test', []).extend(files[second_split:])

    def split(self, split):
        return PCDatasetResampled(self.paths[split], self.num_points, self.labels, **self.augmentation_settings)

    def to_numpy(self):
        original_path = os.path.join(self.data_dir, 'customShapeNet')
        if not os.path.exists(original_path):
            'Download shapenet as in https://github.com/TheoDEPRELLE/AtlasNetV2'
        for code, label in self.labels.items():
            files = glob2.glob(os.path.join(original_path, code, 'ply', '*.ply'))
            shapenet_label_path = os.path.join(self.shapenet_path, label)
            if not os.path.exists(shapenet_label_path):
                os.mkdir(shapenet_label_path)
            i = 0
            for file in files:
                if file.find('*.') > -1:  # Here * is not a wildcard but a character
                    continue
                np_file = os.path.join(shapenet_label_path, str(i))
                if os.path.exists(np_file):
                    i += 1
                    continue
                try:
                    pc = np.loadtxt(file, skiprows=12, usecols=(0, 1, 2))
                except UnicodeDecodeError:
                    print(f'File with path: \n {file} \n is corrupted.')
                else:
                    np.save(np_file, pc)
                finally:
                    i += 1


class DFaustDataset(Dataset):

    def __init__(self, data_dir, k, num_points, rotation, translation):
        self.data_dir = data_dir
        self.rotation = rotation
        self.translation_and_scale = translation
        files = glob2.glob(os.path.join(self.data_dir, 'dfaust', '*'))
        assert files, 'registrations in dataset/dfaust are missing and/or the folder has not been created' \
                      '\nregistrations can be dowloaded from https://dfaust.is.tue.mpg.de/download.php '
        self.pcd, self.indices = load_h5_dfaust(files, k)

    def __len__(self):
        return len(self.pcd)

    def __getitem__(self, index):
        cloud, index = self.pcd[index], self.indices[index]
        cloud = torch.from_numpy(cloud)
        index = torch.from_numpy(index).long()
        if self.rotation:
            cloud, = random_rotation(cloud)
        if self.translation_and_scale:
            cloud, = random_scale_translate(cloud)
        return [cloud, index], 0


class MPIFaustDataset(Dataset):

    def __init__(self, data_dir, k, num_points, rotation, translation):
        self.data_dir = data_dir
        self.rotation = rotation
        self.translation_and_scale = translation
        self.num_points = num_points
        self.files = glob2.glob(os.path.join(self.data_dir, 'MPI-FAUST', 'training', 'registrations', '*.ply'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        mesh = openmesh.read_trimesh(self.files[index])
        cloud = mesh.points().astype(np.float32)
        cloud = torch.from_numpy(cloud)
        if self.rotation:
            cloud, = random_rotation(cloud)
        if self.translation_and_scale:
            cloud, = random_scale_translate(cloud)
        return [cloud, 0], 0


class FaustDataset:

    def __init__(self, data_dir, k, num_points, **augmentation_settings):
        self.data_dir = data_dir
        self.k = k
        self.augmentation_settings = augmentation_settings
        self.num_points = num_points

    def split(self, split):
        assert split in ['trainval', 'test'], 'Dataset only used for testing'
        if split == 'trainval':
            return DFaustDataset(self.data_dir, self.k, self.num_points, **self.augmentation_settings)
        if split == 'test':
            return MPIFaustDataset(self.data_dir, self.k, self.num_points, **self.augmentation_settings)


def get_loaders(dataset_name, batch_size, final, dir_path, **dataset_settings):
    data_dir = os.path.join(dir_path, 'dataset')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    dataset_settings.update(data_dir=data_dir)
    pin_memory = torch.cuda.is_available()
    if dataset_name == 'coins':
        dataset = PiercedCoinsDataset(**dataset_settings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)
        return loader, loader, loader
    if dataset_name == 'modelnet40':
        dataset = Modelnet40Dataset(**dataset_settings)
    elif dataset_name == 'shapenet':
        dataset = ShapeNetDataset(**dataset_settings)
    elif dataset_name == 'shapenet_old':
        dataset = ShapeNetOldDataset(**dataset_settings)
    elif dataset_name == 'faust':
        dataset = FaustDataset(**dataset_settings)
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
        test_dataset, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=pin_memory)
    del dataset
    return train_loader, val_loader, test_loader


def get_cw_loaders(t, final, filter_class=None):
    pin_memory = torch.cuda.is_available()
    batch_size = t.train_loader.batch_size
    t.train_loader.dataset.rotation = False
    t.test(partition='train')
    t.train_loader.dataset.rotation = True
    cw_train_dataset = CWDataset(t.test_outputs['cw_e'], t.test_outputs['cw_idx'], t.test_targets, filter_class)
    t.test(partition='test' if final else 'val')
    cw_test_dataset = CWDataset(t.test_outputs['cw_e'], t.test_outputs['cw_idx'], t.test_targets, filter_class)
    cw_train_loader = torch.utils.data.DataLoader(
        cw_train_dataset, drop_last=True, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    cw_test_loader = torch.utils.data.DataLoader(
        cw_test_dataset, drop_last=False, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return cw_train_loader, cw_test_loader
