import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import re
from sklearn import svm, metrics
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from collections import UserDict
from src.loss_and_metrics import get_ae_loss, CWEncoderLoss, AllMetrics
from src.plot_PC import pc_show
from src.neighbour_op import square_distance
from src.loss_and_metrics import chamfer
from src.loss_and_metrics import emdModule


# Apply recursively lists or dictionaries until check
def apply(obj, check, f):  # changes device in dictionary and lists
    if check(obj):
        return f(obj)
    elif isinstance(obj, list):
        obj = [apply(item, check, f) for item in obj]
    elif isinstance(obj, dict):
        obj = {k: apply(v, check, f) for k, v in obj.items()}
    else:
        raise ValueError(f' Cannot apply {f} on Datatype {type(obj)}')
    return obj


# Dict for (nested) list of Tensor
class TorchDictList(UserDict):

    def __getitem__(self, key_or_index):
        if isinstance(key_or_index, int):
            return self._index_dict_list(key_or_index)
        return super().__getitem__(key_or_index)

    # Indexes a (nested) list in a dictionary
    def _index_dict_list(self, ind):
        out_dict = {}
        for k, v in self.items():
            if not v or isinstance(v[0], list):
                new_v = [elem[ind].unsqueeze(0) for elem in v]
            else:
                new_v = v[ind].unsqueeze(0)
            out_dict[k] = new_v
        return out_dict

    # Separates batch into list and appends (or creates) to structure dict of (nested) lists
    def extend_dict(self, new_dict):
        for key, value in new_dict.items():
            if isinstance(value, list):
                for elem, new_elem in zip(self.setdefault(key, [[] for _ in value]), value):
                    assert torch.is_tensor(new_elem)
                    elem.extend(new_elem)
            else:
                assert torch.is_tensor(value)
                self.setdefault(key, []).extend(value)


'''
This abstract class manages training and general utilities.
The outputs from the network are saved in a dictionary and stored in a list.
The dictionary also handles list of tensors as values.


The loss is an abstract method later defined and returns a dictionary dict with 
the different components of the criterion, plus eventually other useful metrics. 
The loss is supposed to be already averaged on the batch and the epoch loss is slightly off 
whenever the last batch is smaller. For large dataset and small batch size the error should be irrelevant.

dict['Criterion'] = loss to backprop


To save and load on a separate sever, it can handle a Minio object from the minio library.
This object downloads and uploads the model to a separate storage.
In order to get full access to this class utilities, set up MinIO on your storage device (https://min.io)
Install the minio api and pass the minioClient:

from minio import Minio
minioClient = Minio(*Your storage name*,
                  access_key= *Your access key*,
                  secret_key= *Your secret key*,
                  secure=True)

'''


class Trainer(metaclass=ABCMeta):
    bin = ''
    quiet_mode = False  # less output
    max_output = np.inf  # maximum amount of stored evaluated test samples

    def __init__(self, model, exp_name, device, optimizer, train_loader, val_loader=None,
                 test_loader=None, minio_client=None, models_path='./models', **block_args):

        self.epoch = 0
        self.device = device  # to cuda or not to cuda?
        self.model = model.to(device)
        self.exp_name = exp_name  # name used for saving and loading
        self.schedule = block_args['schedule']
        self.settings = {**model.settings, **block_args}
        self.optimizer_settings = block_args['optim_args'].copy()
        self.optimizer = optimizer(**self.optimizer_settings)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_losses, self.val_losses, self.test_losses = {}, {}, {}
        self.converge = 1  # if 0 kills session
        self.minio = minio_client
        self.models_path = models_path
        self.minio_path = staticmethod(lambda path: path[len(models_path) + 1:]).__func__  # removes model path
        self.test_indices, self.test_targets, self.test_outputs = None, None, None  # store last test evaluation
        self.saved_metrics = {}  # saves metrics of last evaluation
        settings_path = self.paths()['settings']
        json.dump(self.settings, open(settings_path, 'w'), default=vars, indent=4)

    @property
    def optimizer_settings(self):  # settings shown depend on epoch
        if self.schedule is None:
            return {'params': self._optimizer_settings[0],
                    **self._optimizer_settings[1]}
        else:  # the scheduler modifies the learning rate(s)
            init_learning = self._optimizer_settings[0]
            scheduled_learning = []
            for group in init_learning:
                scheduled_learning.append({
                    'params': group['params'],
                    'lr': self.schedule(group['lr'], self.epoch)
                })
            return {'params': scheduled_learning,
                    **self._optimizer_settings[1]}

    @optimizer_settings.setter
    def optimizer_settings(self, optim_args):
        lr = optim_args.pop('lr')
        if isinstance(lr, dict):  # support individual lr for each parameter (for finetuning for example)
            self._optimizer_settings = \
                [{'params': getattr(self.model, k).parameters(), 'lr': v} for k, v in lr.items()], optim_args
        else:
            self._optimizer_settings = [{'params': self.model.parameters(), 'lr': lr}], optim_args
        return

    def update_learning_rate(self, new_lr):
        if not isinstance(new_lr, list):  # transform to list
            new_lr = [{'lr': new_lr} for _ in self.optimizer.param_groups]
        for g, up_g in zip(self.optimizer.param_groups, new_lr):
            g['lr'] = up_g['lr']
        return

    def train(self, num_epoch, val_after_train=False):
        if not self.quiet_mode:
            print('Experiment name ', self.exp_name)
        for _ in range(num_epoch):
            self.update_learning_rate(self.optimizer_settings['params'])
            self.epoch += 1
            if self.quiet_mode:
                print('\r====> Epoch:{:3d}'.format(self.epoch), end='')
            else:
                print('====> Epoch:{:3d}'.format(self.epoch))
            self._run_session(partition='train')
            if self.val_loader and val_after_train:  # check losses on val
                self._run_session(partition='val', inference=True)
        return

    def test(self, partition, save_outputs=0, **kwargs):  # runs and stores evaluated test samples
        save_outputs = self.max_output if save_outputs else 0
        if not self.quiet_mode:
            print('Version ', self.exp_name)
        self._run_session(partition=partition, inference=True, save_outputs=save_outputs)
        return

    def _run_session(self, partition='train', inference=False, save_outputs=0):
        if inference:
            self.model.eval()
            torch.set_grad_enabled(False)
        else:
            self.model.train()
            torch.set_grad_enabled(True)
        if partition == 'train':
            loader = self.train_loader
            dict_losses = self.train_losses
        elif partition == 'val':
            loader = self.val_loader
            dict_losses = self.val_losses
        elif partition == 'test':
            loader = self.test_loader
            dict_losses = self.test_losses
        else:
            raise ValueError('partition options are: "train", "val", "test" ')
        if save_outputs:
            self.test_indices, self.test_targets, self.test_outputs = [], [], TorchDictList()

        epoch_loss = {}
        epoch_metrics = {}
        num_batch = len(loader)
        iterable = tqdm(enumerate(loader), total=num_batch, disable=self.quiet_mode)
        epoch_seen = 0
        for batch_idx, (inputs, targets, indices) in iterable:
            if self.converge == 0:
                return
            epoch_seen += indices.shape[0]
            inputs, targets = self.recursive_to([inputs, targets], self.device)
            inputs_aux = self.helper_inputs(inputs, targets)
            outputs = self.model(**inputs_aux)
            if inference:
                batch_metrics = self.metrics(outputs, inputs, targets)
                for metric, value in batch_metrics.items():
                    epoch_metrics[metric] = epoch_metrics.get(metric, 0) + value.item()
            else:
                batch_loss = self.loss(outputs, inputs, targets)
                criterion = batch_loss['Criterion']
                assert not torch.isnan(criterion), print('NANs found:', outputs)
                for loss, value in batch_loss.items():
                    epoch_loss[loss] = epoch_loss.get(loss, 0) + value.item()
                if torch.isinf(criterion) or torch.isnan(criterion):
                    self.converge = 0
                criterion.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if not self.quiet_mode:
                    if batch_idx % (num_batch // 10 or 1) == 0:
                        iterable.set_postfix({'Seen': epoch_seen,
                                              'Loss': criterion.item()})
                    if batch_idx == num_batch - 1:  # clear after last
                        iterable.set_description('')
            if save_outputs > (batch_idx + 1) * loader.batch_size:
                self.test_outputs.extend_dict(self.recursive_to(outputs, 'detach_cpu'))
                self.test_indices.extend(indices)
                self.test_targets.extend(self.recursive_to(targets, 'detach_cpu'))
        if inference:  # not averaged in batch
            self.saved_metrics = {metric: value / num_batch if metric == 'Criterion' else value / epoch_seen
                                  for metric, value in epoch_metrics.items()}
            print('Metrics:')
            for metric, value in self.saved_metrics.items():
                print('{}: {:.4e}'.format(metric, value), end='\t')
            print()
        else:
            epoch_loss = {loss: value / num_batch if loss == 'Criterion' else value / epoch_seen
                          for loss, value in epoch_loss.items()}
            for loss, value in epoch_loss.items():
                dict_losses.setdefault(loss, []).append(value)
            if not self.quiet_mode:
                print('Average {} losses :'.format(partition))
                for loss, value in epoch_loss.items():
                    print('{}: {:.4f}'.format(loss, value), end='\t')
                print()
        return

    @abstractmethod
    def loss(self, output, inputs, targets):
        pass

    def metrics(self, output, inputs, targets):
        return self.loss(output, inputs, targets)

    def helper_inputs(self, inputs, labels):
        return {'x': inputs}

    def plot_losses(self, loss):
        tidy_loss = ' '.join([s.capitalize() for s in loss.split('_')])
        epochs = np.arange(self.epoch)
        plt.plot(epochs, self.train_losses[loss], label='train')
        if self.val_loader:
            plt.plot(epochs, self.val_losses[loss], label='val')
        plt.xlabel('Epochs')
        plt.ylabel(tidy_loss)
        plt.title(f'{self.exp_name}')
        plt.show()
        return

    # Change device recursively to tensors inside a list or a dictionary
    @staticmethod
    def recursive_to(obj, device):  # changes device in dictionary and lists
        if device == 'detach_cpu':
            return apply(obj, check=torch.is_tensor, f=lambda x: x.detach().cpu())
        return apply(obj, check=torch.is_tensor, f=lambda x: x.to(device))

    def save(self, new_exp_name=None):
        self.model.eval()
        paths = self.paths(new_exp_name)
        torch.save(self.model.state_dict(), paths['model'])
        torch.save(self.optimizer.state_dict(), paths['optim'])
        json.dump(self.train_losses, open(paths['train_hist'], 'w'))
        json.dump(self.val_losses, open(paths['val_hist'], 'w'))
        if new_exp_name:
            json.dump(self.settings, open(paths['settings'], 'w'))
        if self.minio is not None:
            for file in paths.values():
                self.minio.fput_object(self.bin, self.minio_path(file), file)
        print('Model saved at: ', paths['model'])
        return

    # looks on the server (using minio) first, then on the local storage
    def load(self, epoch=None):
        directory = self.exp_name

        if epoch is not None:
            self.epoch = epoch
        else:
            past_epochs = []  # here it looks for the most recent model
            if self.minio is not None:
                for file in self.minio.list_objects(self.bin, recursive=True):
                    file_dir, *file_name = file.object_name.split('/')
                    if file_dir == directory and file_name[0][:5] == 'model':
                        past_epochs.append(int(re.sub('\D', '', file_name[0])))
            local_path = os.path.join(self.models_path, self.exp_name)
            if os.path.exists(local_path):
                for file in os.listdir(local_path):
                    if file[:5] == 'model':
                        past_epochs.append(int(re.sub('\D', '', file)))
            if not past_epochs:
                print('No saved models found')
                return
            else:
                self.epoch = max(past_epochs)
        paths = self.paths()
        if self.minio is not None:
            for file in paths.values():
                if file[-13:] != 'settings.json':
                    self.minio.fget_object(self.bin, self.minio_path(file), file)
        self.model.load_state_dict(torch.load(paths['model'],
                                              map_location=torch.device(self.device)))
        self.optimizer.load_state_dict(torch.load(paths['optim'],
                                                  map_location=torch.device(self.device)))
        self.train_losses = json.load(open(paths['train_hist']))
        self.val_losses = json.load(open(paths['val_hist']))
        print('Loaded: ', paths['model'])
        return

    def paths(self, new_exp_name=None, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)
        if new_exp_name:  # save a parallel version to work with
            directory = os.path.join(self.models_path, new_exp_name)
        else:
            directory = os.path.join(self.models_path, self.exp_name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        paths = {'settings': os.path.join(directory, 'settings.json'),
                 'model': os.path.join(directory, f'model_epoch{epoch}.pt'),
                 'optim': os.path.join(directory, f'optimizer_epoch{epoch}.pt'),
                 'train_hist': os.path.join(directory, 'train_losses.json'),
                 'val_hist': os.path.join(directory, 'val_losses.json')}
        return paths


class AETrainer(Trainer):
    clf = svm.SVC(kernel='linear')
    bin = 'pcdvae'  # minio bin
    saved_accuracies = {}

    def __init__(self, model, exp_name, block_args):
        super().__init__(model, exp_name, **block_args)
        self.acc = None
        self.cf = None  # confusion matrix
        self._loss = get_ae_loss(block_args)
        self._metrics = self._loss
        return

    def test(self, partition, all_metrics=False, denormalise=False, save_outputs=0, **kwargs):
        m_old = self.model.decoder.m
        self.model.decoder.m = kwargs['m']
        if all_metrics:
            self._metrics = lambda x, y, z: AllMetrics(denormalise)(x, y)
        super().test(partition=partition, save_outputs=save_outputs)
        self.model.decoder.m = m_old
        return

    def test_cw_recon(self, *args, **kwargs):
        try:
            self.model.recon_cw = True
        except AttributeError:
            print('Codeword recontruction is only supported for the VQVAE model')
        else:
            self.test(*args, **kwargs)
            self.model.recon_cw = False
        return

    def update_m_training(self, m):
        self.model.decoder.m_training = m

    def clas_metric(self, final=False):
        # No rotation here
        self.train_loader.dataset.rotation = False
        self.test(partition='train', save_outputs=self.test_outputs)
        self.train_loader.dataset.rotation = True
        x_train = np.array([cw.numpy() for cw in self.test_outputs['cw']])
        y_train = np.array([cw.numpy() for cw in self.test_targets])
        shuffle = np.random.permutation(y_train.shape[0])
        x_train = x_train[shuffle]
        y_train = y_train[shuffle]
        print('Fitting the classifier ...')
        self.clf.fit(x_train, y_train)
        partition = 'test' if final else 'val'
        self.test(partition=partition)
        x_test = np.array([cw.numpy() for cw in self.test_outputs['cw']])
        y_test = np.array([cw.numpy() for cw in self.test_targets])
        y_hat = self.clf.predict(x_test)
        self.acc = (y_hat == y_test).sum() / y_hat.shape[0]
        print('Accuracy: ', self.acc)
        self.cf = metrics.confusion_matrix(y_hat, y_test, normalize='true')
        print('Mean Accuracy;', np.diag(self.cf).astype(float).mean())
        directory = os.path.join(self.models_path, self.exp_name)
        accuracy_path = os.path.join(directory, 'svm_accuracies.json')
        self.saved_accuracies[self.epoch] = self.acc
        json.dump(self.saved_accuracies, open(accuracy_path, 'w'))
        return self.acc

    def latent_visualisation(self, highlight_label):
        from sklearn.decomposition import PCA
        cw = torch.stack(self.test_outputs['cw'])
        pca = PCA(3)
        cw_pca = pca.fit_transform(cw.numpy())
        labels = torch.stack(self.test_targets).cpu().numpy()
        highlight_cw = cw_pca[(highlight_label == labels)]
        pc_show([torch.FloatTensor(cw_pca), highlight_cw], colors=['blue', 'red'])

    def loss(self, output, inputs, targets):
        return self._loss(output, inputs, targets)

    def helper_inputs(self, inputs, labels):
        # inputs length vary on the dataset, when resampling two different resamplings of the shape are given
        indices = inputs[-1]
        if torch.all(indices == 0):
            indices = None
        input_shape = inputs[1]
        return {'x': input_shape, 'indices': indices}

    def metrics(self, output, inputs, targets):
        return self._metrics(output, inputs, targets)

    def evaluate_generated_set(self, m, metric='Chamfer'):
        self.model.decoder.m = m
        test_dataset = []
        generated_dataset = []
        for batch_idx, (inputs, targets, index) in enumerate(self.val_loader):
            test_clouds = inputs[1]
            test_dataset.extend(test_clouds.cpu())
            # (samples, targets, index) = next(iter(self.train_loader))
            # samples = samples[1]
            self.model.eval()
            with torch.no_grad():
                samples = self.model.random_sampling(test_clouds.shape[0])['recon'].detach().cpu()
            generated_dataset.extend(samples)
        test_l = len(test_dataset)
        dist_array = np.zeros((2 * test_l, 2 * test_l), dtype=float)
        all_shapes = test_dataset + generated_dataset[:test_l]
        emd = emdModule()
        for i, shapei in enumerate(all_shapes):
            for j, shapej in enumerate(all_shapes):
                if i == j:
                    dist_array[i, j] = np.inf  # safe way of ignoring this entry
                elif i > j:  # distance is symmetric

                    cloud1 = shapei.unsqueeze(0).to(self.device)
                    cloud2 = shapej.unsqueeze(0).to(self.device)
                    if metric == 'Chamfer':
                        pairwise_dist = square_distance(cloud1, cloud2)
                        ch = chamfer(cloud1, cloud2, pairwise_dist)[0].sum(0) / cloud1.shape[1]
                        dist = ch.item()
                    else:
                        assert metric == 'Emd'
                        dist = emd(cloud1, cloud2, 0.05, 3000)[0].mean()
                    dist_array[i, j] = dist
                    dist_array[j, i] = dist
        np.save('dist_array', dist_array)
        closest = dist_array.argmin(axis=1)  # np.inf on the diagonal so argmin != index
        test_closest = closest[:test_l]
        generated_closest = closest[test_l:]
        nna = ((test_closest < test_l).sum() + (generated_closest >= test_l).sum()) / (2 * test_l)
        coverage_array = dist_array[test_l:, :test_l]
        coverage_closest = coverage_array.argmin(axis=1)
        coverage = np.unique(coverage_closest).shape[0] / test_l
        mmd_array = dist_array[:test_l, test_l:]
        mmd = mmd_array.min(axis=1).mean()
        print(f'Coverage score ({metric}): {coverage:.4e}')
        print(f'Minimum Matching Distance score ({metric}): {mmd:.4e}')
        print(f'1-NNA score ({metric}): {nna:.4e}')
        return dist_array



class CWTrainer(Trainer):

    def __init__(self, model, exp_name, block_args):
        self.vqvae_model = model
        self.vqvae_epoch = block_args['training_epochs']
        super().__init__(model.cw_encoder, exp_name, **block_args)
        self._loss = CWEncoderLoss(block_args['c_reg'])
        self.model.codebook = torch.nn.Parameter(self.vqvae_model.codebook.detach().clone(), requires_grad=False)
        return

    def loss(self, output, inputs, targets):
        return self._loss(output, inputs, targets)

    def save(self, new_exp_name=None):
        self.model.eval()
        paths = self.paths(new_exp_name, epoch=self.vqvae_epoch)
        self.vqvae_model.cw_encoder.load_state_dict(self.model.state_dict())
        if new_exp_name:
            json.dump(self.settings, open(paths['settings'], 'w'))
        torch.save(self.vqvae_model.state_dict(), paths['model'])
        if self.minio is not None:
            self.minio.fput_object(self.bin, self.minio_path(paths['model']), paths['model'])
        print('Model saved at: ', paths['model'])
        return

    def helper_inputs(self, inputs, labels):
        return {'x': inputs[0]}




def get_ae_trainer(model, exp_name, block_args):
    return AETrainer(model, exp_name, block_args)
