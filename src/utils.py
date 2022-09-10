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



