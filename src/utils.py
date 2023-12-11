import os
import zipfile
import numpy as np
import requests
import h5py
import glob2
from sklearn.neighbors import KDTree

try:
    from requests.packages.urllib3.exceptions import InsecureRequestWarning

    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
except ImportError:
    InsecureRequestWarning = None
    pass


def download_zip(data_dir, zip_path, url):
    if not os.path.exists(zip_path):
        r = requests.get(url, verify=False)
        open(zip_path, 'wb').write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    return


def index_k_neighbours(pcs, k):
    indices_list = []
    for pc in pcs:
        kdtree = KDTree(pc)
        indices = kdtree.query(pc, k, return_distance=False)
        indices_list.append(indices.reshape(-1, k))
    return np.stack(indices_list, axis=0)


def load_h5_modelnet(wild_path, input_points, k):
    pcd = []
    indices = []
    labels = []
    for h5_name in glob2.glob(wild_path):
        with h5py.File(h5_name, 'r+') as f:
            print('Load: ', h5_name)
            # Dataset is already normalized
            pcs = f['data'][:].astype('float32')
            pcs = pcs[:, :input_points, :]
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


def load_h5_dfaust(files, input_points, k):
    clouds = []
    indices = []
    for file_path in files:
        print('Load: ', file_path)
        clouds_file = []
        indices_file = []
        with h5py.File(file_path, 'r+') as f:
            for name, dset in f.items():
                if name == 'faces' or name.find('index') > -1:
                    continue
                pc = dset[:][:, :input_points, :].transpose((2, 0, 1))
                index_k = f'{name}_index_{k}'
                if index_k in f.keys():
                    index = f[index_k][:].astype(np.short)
                else:
                    del f[index_k]
                    index = index_k_neighbours(pc, k).astype(np.short)
                    f.create_dataset(index_k, data=index)
                clouds_file.extend(pc)
                indices_file.extend(index)
        clouds.extend(clouds_file)
        indices.extend(indices_file)
    return clouds, indices


# Allows a temporary change using the with statement
class UsuallyFalse:
    _value = False

    def __bool__(self):
        return self._value

    def __enter__(self):
        self._value = True

    def __exit__(self, *_):
        self._value = False
