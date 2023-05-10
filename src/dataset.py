import numpy as np
import torch
import os
from torch.utils.data import Dataset
from src.utils import download_zip, index_k_neighbours, load_h5_modelnet, load_h5_dfaust
import json
import glob2
import openmesh


def normalise(cloud):
    cloud -= cloud.mean(axis=0)
    std = np.max(np.sqrt(np.sum(cloud ** 2, axis=1)))
    cloud /= std
    return cloud, std


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


class EmptyDataset(Dataset):

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return NotImplementedError


class CWDataset(Dataset):
    def __init__(self, cw_e, cw_idx, labels, **other_settings):
        self.cw_e = torch.stack(cw_e)
        self.cw_idx = torch.stack(cw_idx)
        self.labels = torch.stack(labels)

    def __len__(self):
        return self.cw_e.shape[0]

    def __getitem__(self, index):
        return [self.cw_e[index], self.cw_idx[index]], self.labels[index], index


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
        return [1, plate, self.neighbours], n_holes, index


class AugmentDataset(Dataset):
    def __init__(self, rotation, translation, **kwargs):
        self.rotation = rotation
        self.translation_and_scale = translation

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        return NotImplementedError

    def augment(self, clouds):
        if self.rotation:
            clouds = random_rotation(*clouds)
        if self.translation_and_scale:
            clouds = random_scale_translate(*clouds)
        return clouds


class Modelnet40Split(AugmentDataset):
    def __init__(self, pcd, indices, labels, rotation, translation, **other_settings):
        super().__init__(rotation, translation)
        self.pcd = pcd.astype(np.float32)
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return self.pcd.shape[0]

    def __getitem__(self, index):
        cloud, neighbours_indices, label = self.pcd[index], self.indices[index], self.labels[index]
        cloud = torch.from_numpy(cloud)
        neighbours_indices = torch.from_numpy(neighbours_indices).long()
        clouds = self.augment([cloud])
        return [1., *clouds, neighbours_indices], label, index


class ShapenetAtlasSplit(AugmentDataset):
    def __init__(self, paths, input_points, labels, resample, rotation, translation, **other_settings):
        super().__init__(rotation, translation)
        self.paths = paths
        self.translation_and_scale = translation
        self.input_points = input_points
        self.resample = resample
        self.label_index = list(labels.values())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        cloud = np.load(path).astype(np.float32)
        index_pool = np.arange(cloud.shape[0])
        sampling = np.random.choice(index_pool, size=(1 + self.resample) * self.input_points, replace=self.resample)
        ref_cloud, scale = normalise(cloud[sampling[:self.input_points]])
        clouds = [torch.from_numpy(ref_cloud)]
        if self.resample:
            clouds.append(torch.from_numpy(normalise(cloud[sampling[self.input_points:]])[0]))
        label = path.split(os.sep)[-2]
        label = self.label_index.index(label)
        clouds = self.augment(clouds)
        return [scale, *clouds, 0], label, index


class ShapenetFlowSplit(AugmentDataset):
    def __init__(self, paths, input_points, labels, resample, rotation, translation, **other_settings):
        super().__init__(rotation, translation)
        self.paths = paths
        self.pcd = []
        self.labels = []
        self.resample = resample
        self.input_points = input_points
        self.label_index = list(labels.keys())
        self.scales = []
        for path in paths:
            pc, scale = normalise(np.load(path))
            self.pcd.append(pc.astype(np.float32))
            self.scales.append(scale)
            label = path.split(os.sep)[-3]
            self.labels.append(self.label_index.index(label))

    def __len__(self):
        return len(self.pcd)

    def __getitem__(self, index):
        cloud = self.pcd[index]
        label = self.labels[index]
        scale = self.scales[index]
        index_pool = np.arange(cloud.shape[0])
        sampling = np.random.choice(index_pool, size=(1 + self.resample) * self.input_points, replace=True)
        clouds = [torch.from_numpy(cloud[sampling[:self.input_points]])]
        if self.resample:
            clouds.append(torch.from_numpy(cloud[sampling[self.input_points:]]))
        clouds = self.augment(clouds)
        return [scale, *clouds, 0], label, index


class Modelnet40Dataset:
    with open(os.path.join('metadata', 'modelnet_classes.txt'), 'r') as f:
        classes = f.read().splitlines()

    def __init__(self, data_dir, k, input_points, select_classes, **augmentation_settings):
        self.data_dir = data_dir
        self.modelnet_path = os.path.join(data_dir, 'modelnet40_hdf5_2048')
        self.augmentation_settings = augmentation_settings
        self.download()
        self.pcd, self.indices, self.labels = {}, {}, {}
        for split in ['train', 'test']:
            self.pcd[split], self.indices[split], self.labels[split] = \
                load_h5_modelnet(os.path.join(self.modelnet_path, f'*{split}*.h5'), input_points, k)
            if select_classes:
                try:
                    selected_labels = [self.classes.index(selected_class) for selected_class in select_classes]
                except ValueError:
                    print(f'One of classes in {select_classes} not in {split} dataset')
                    raise
                selected_indices = np.isin(self.labels[split], selected_labels)
                self.pcd[split] = self.pcd[split][selected_indices]
                self.indices[split] = self.indices[split][selected_indices]
                self.labels[split] = self.labels[split][selected_indices]

    def split(self, split):
        if split == 'train_val':
            assert 'val' not in self.pcd.keys(), 'train dataset has already been split'
            split = 'train'
        elif split in ['train', 'val'] and 'val' not in self.pcd.keys():
            self.train_val_to_train_and_val()
        return Modelnet40Split(pcd=self.pcd[split], indices=self.indices[split],
                               labels=self.labels[split], **self.augmentation_settings)

    def download(self):
        url = 'https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097/?dl=1'
        return download_zip(data_dir=self.data_dir, zip_path=self.modelnet_path + '.zip', url=url)

    def train_val_to_train_and_val(self, val_every=6):
        train_idx = list(range(self.pcd['train'].shape[0]))
        val_idx = [train_idx.pop(i) for i in train_idx[::-val_every]]
        # partition train into train and val
        for new_split, new_split_idx in [['val', val_idx], ['train', train_idx]]:
            self.pcd[new_split] = self.pcd['train'][new_split_idx]
            self.indices[new_split] = self.indices['train'][new_split_idx]
            self.labels[new_split] = self.labels['train'][new_split_idx]


class ShapeNetDatasetAtlas:
    classes = json.load(open(os.path.join(os.getcwd(), 'metadata', 'shapenet_AtlasNet_classes.json'), 'r'))

    def __init__(self, data_dir, k, input_points, select_classes, **augmentation_settings):
        self.data_dir = data_dir
        self.shapenet_path = os.path.join(self.data_dir, 'shapenet')
        if not os.path.exists(self.shapenet_path):
            os.mkdir(self.shapenet_path)
            self.to_numpy()
        self.val_ratio = 0.2
        self.test_ratio = 0.2
        self.augmentation_settings = augmentation_settings
        self.input_points = input_points
        folders = glob2.glob(os.path.join(self.shapenet_path, '*'))
        self.paths = {}
        if select_classes:
            folders = [folder for folder in folders if os.path.split(folder)[1] in select_classes]
            assert folders, 'class is not in dataset'
        for folder in folders:
            files = glob2.glob(os.path.join(folder, '*'))
            first_split = int(len(files) * (1 - self.val_ratio - self.test_ratio))
            second_split = int(len(files) * (1 - self.test_ratio))
            self.paths.setdefault('train', []).extend(files[:first_split])
            self.paths.setdefault('train_val', []).extend(files[:second_split])
            self.paths.setdefault('val', []).extend(files[first_split:second_split])
            self.paths.setdefault('test', []).extend(files[second_split:])

    def split(self, split):
        return ShapenetAtlasSplit(self.paths[split], self.input_points, self.classes, resample=split in ['val', 'test'],
                                  **self.augmentation_settings)

    def to_numpy(self):
        original_path = os.path.join(self.data_dir, 'customShapeNet')
        if not os.path.exists(original_path):
            'Download shapenet as in https://github.com/TheoDEPRELLE/AtlasNetV2'
        for code, label in self.classes.items():
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


class ShapeNetDatasetFlow:
    classes = json.load(open(os.path.join(os.getcwd(), 'metadata', 'shapenet_PointFlow_classes.json'), 'r'))

    def __init__(self, data_dir, k, input_points, select_classes, **augmentation_settings):
        self.data_dir = data_dir
        self.shapenet_path = os.path.join(self.data_dir, 'ShapeNetCore.v2.PC15k')
        link = 'https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ'
        assert os.path.exists(self.shapenet_path), f'Download and extract dataset from here: {link}'
        self.augmentation_settings = augmentation_settings
        self.input_points = input_points
        folders = glob2.glob(os.path.join(self.shapenet_path, '*'))
        self.paths = {}
        if select_classes:
            folders = [folder for folder in folders if self.classes[os.path.split(folder)[1]] in select_classes]
            assert folders, 'class is not in dataset'
        for folder in folders:
            train_files = glob2.glob(os.path.join(folder, 'train', '*'))
            val_files = glob2.glob(os.path.join(folder, 'val', '*'))
            test_files = glob2.glob(os.path.join(folder, 'test', '*'))
            train_val_files = train_files + val_files
            self.paths.setdefault('train', []).extend(train_files)
            self.paths.setdefault('train_val', []).extend(train_val_files)
            self.paths.setdefault('val', []).extend(val_files)
            self.paths.setdefault('test', []).extend(test_files)

    def split(self, split):
        return ShapenetFlowSplit(self.paths[split], self.input_points, self.classes,
                                 **self.augmentation_settings)


class DFaustDataset(Dataset):

    def __init__(self, data_dir, k, input_points, rotation, translation):
        self.data_dir = data_dir
        self.rotation = rotation
        self.translation_and_scale = translation
        files = glob2.glob(os.path.join(self.data_dir, 'dfaust', '*'))
        assert files, 'registrations are missing and/or the folder has not been created' \
                      '\nregistrations can be downloaded from https://dfaust.is.tue.mpg.de/download.php '
        self.pcd, self.indices = load_h5_dfaust(files, k)

    def __len__(self):
        return len(self.pcd)

    def __getitem__(self, index):
        cloud, index = self.pcd[index], self.indices[index]
        cloud, scale = normalise(cloud)
        cloud = torch.from_numpy(cloud)
        neighbours_indices = torch.from_numpy(index).long()
        if self.rotation:
            cloud, = random_rotation(cloud)
        if self.translation_and_scale:
            cloud, = random_scale_translate(cloud)
        return [scale, cloud, neighbours_indices], 0, index


class MPIFaustDataset(Dataset):

    def __init__(self, data_dir, input_points, rotation, translation):
        self.data_dir = data_dir
        self.rotation = rotation
        self.translation_and_scale = translation
        self.input_points = input_points
        self.files = glob2.glob(os.path.join(self.data_dir, 'MPI-FAUST', 'training', 'registrations', '*.ply'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        mesh = openmesh.read_trimesh(self.files[index])
        cloud = mesh.points().astype(np.float32)
        cloud = torch.from_numpy(normalise(cloud))
        if self.rotation:
            cloud, = random_rotation(cloud)
        if self.translation_and_scale:
            cloud, = random_scale_translate(cloud)
        return [cloud, 0], 0


class FaustDataset:

    def __init__(self, data_dir, k, input_points, **augmentation_settings):
        self.data_dir = data_dir
        self.k = k
        self.augmentation_settings = augmentation_settings
        self.input_points = input_points

    def split(self, split):
        assert split in ['train_val', 'test'], 'Dataset only used for testing'
        if split == 'train_val':
            return DFaustDataset(self.data_dir, self.k, self.input_points, **self.augmentation_settings)
        if split == 'test':
            return MPIFaustDataset(self.data_dir, self.k, self.input_points, **self.augmentation_settings)


def get_dataset(dataset_name):
    dataset_dict = {'Coins': PiercedCoinsDataset,
                    'Modelnet40': Modelnet40Dataset,
                    'ShapenetAtlas': ShapeNetDatasetAtlas,
                    'ShapenetFlow': ShapeNetDatasetFlow,
                    'Faust': FaustDataset}
    return dataset_dict[dataset_name]


def get_loaders(dataset_name, batch_size, final, data_dir, **dataset_settings):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    dataset_settings.update(data_dir=data_dir)
    pin_memory = torch.cuda.is_available()
    dataset = get_dataset(dataset_name)(**dataset_settings)
    if final:
        train_dataset = dataset.split('train_val')
        train_loader = torch.utils.data.DataLoader(
            train_dataset, drop_last=True, batch_size=batch_size,
            shuffle=True, pin_memory=pin_memory)
        val_loader = None
        test_dataset = dataset.split('test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=pin_memory)
    else:
        train_dataset = dataset.split('train')
        val_dataset = dataset.split('val')
        train_loader = torch.utils.data.DataLoader(
            train_dataset, drop_last=True, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, drop_last=False, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
        test_loader = None

    del dataset
    return train_loader, val_loader, test_loader


def get_cw_loaders(t, partition, batch_size):
    pin_memory = torch.cuda.is_available()
    # m=4 because we don't care about the reconstruction and m < 4 creates problems with the filtering
    t.test(partition='train', m=4, save_outputs=True)
    cw_train_dataset = CWDataset(t.test_outputs['cw_q'], t.test_outputs['cw_idx'], t.test_metadata['targets'])
    t.test(partition=partition, m=4, save_outputs=True)
    cw_test_dataset = CWDataset(t.test_outputs['cw_q'], t.test_outputs['cw_idx'], t.test_metadata['targets'])
    cw_train_loader = torch.utils.data.DataLoader(
        cw_train_dataset, drop_last=True, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    cw_test_loader = torch.utils.data.DataLoader(
        cw_test_dataset, drop_last=False, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    return cw_train_loader, cw_test_loader
