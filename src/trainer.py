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
from src.losses import get_vae_loss
from src.plot_PC import pc_show

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
    quiet_mode = False  # less output
    max_output = np.inf  # maximum amount of stored evaluated test samples

    def __init__(self, model, exp_name, device, optimizer, train_loader, val_loader=None,
                 test_loader=None, minioClient=None, models_path='./models', **block_args):

        self.epoch = 0
        self.device = device  # to cuda or not to cuda?
        self.model = model.to(device)  # model is not copied
        self.exp_name = exp_name  # name used for saving and loading
        self.schedule = block_args['schedule']
        self.settings = {**model.settings, **block_args}
        self.optimizer_settings = block_args['optim_args'].copy()
        self.optimizer = optimizer(**self.optimizer_settings)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_losses, self.val_losses, self.test_losses = {}, {}, {}
        self.test_targets, self.test_outputs = [], {}
        self.converge = 1  # if 0 kills session
        self.minio = minioClient
        self.models_path = models_path
        self.minio_path = staticmethod(lambda path: path[len(models_path+1):]).__func__  # removes model path
        self.test_targets, self.test_outputs = [], {}  # stored in RAM
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
        if isinstance(lr, dict):
            self._optimizer_settings = [
                                           {'params': getattr(self.model, k).parameters(),
                                            'lr': v} for k, v in lr.items()], optim_args
        else:
            self._optimizer_settings = \
                [{'params': self.model.parameters(), 'lr': lr}], optim_args
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
                self._run_session(partition='val', inference=True)  # best to test instead if interested in metrics
        return

    def test(self, partition='val'):  # runs and stores evaluated test samples
        if not self.quiet_mode:
            print('Version ', self.exp_name)
        self._run_session(partition=partition, inference=True, save_outputs=True)
        return

    def _run_session(self, partition='train', inference=False, save_outputs=False):
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
            self.test_targets, self.test_outputs = [], {}

        epoch_loss = {}
        num_batch = len(loader)
        iterable = tqdm(enumerate(loader), total=num_batch, disable=self.quiet_mode)
        for batch_idx, (inputs, targets) in iterable:
            if self.converge == 0:
                return
            inputs, targets = self.to_recursive([inputs, targets], self.device)
            inputs_aux = self.helper_inputs(inputs, targets)
            outputs = self.model(**inputs_aux)
            batch_loss = self.loss(outputs, inputs, targets)
            criterion = batch_loss['Criterion']
            for loss, value in batch_loss.items():
                epoch_loss[loss] = epoch_loss.get(loss, 0) + value.item()
            if not inference:
                if torch.isinf(criterion) or torch.isnan(criterion):
                    self.converge = 0
                criterion.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if save_outputs and \
                    self.max_output > (batch_idx + 1) * loader.batch_size:
                self.extend_dict_list(
                    self.test_outputs, self.to_recursive(outputs, 'detach_cpu'))
                self.test_targets.extend(self.to_recursive(targets, 'detach_cpu'))
            if not self.quiet_mode and partition == 'train':
                if batch_idx % (num_batch // 10 or 1) == 0:
                    iterable.set_postfix({'Seen': batch_idx * loader.batch_size,
                                          'Loss': criterion.item()})
                if batch_idx == num_batch - 1:  # clear after last
                    iterable.set_description('')

        epoch_loss = {loss: value / num_batch for loss, value in epoch_loss.items()}
        if not save_outputs:
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
    def to_recursive(obj, device):  # changes device in dictionary and lists
        if isinstance(obj, list):
            obj = [Trainer.to_recursive(item, device) for item in obj]
        elif isinstance(obj, dict):
            obj = {k: Trainer.to_recursive(v, device) for k, v in obj.items()}
        else:
            try:
                obj = obj.detach().cpu() if device == 'detach_cpu' else obj.to(device)
            except AttributeError:
                raise ValueError(f'Datatype {type(obj)} does not contain tensors')
        return obj

    # Separates batch into list and appends to (creates) structure (dict of lists, dict of lists of lists)
    @staticmethod  # extends lists in dictionaries
    def extend_dict_list(old_dict, new_dict):
        for key, value in new_dict.items():
            if isinstance(value, list):
                for elem, new_elem in zip(old_dict.setdefault(key, [[]] * len(value)), value):
                    assert torch.is_tensor(new_elem)
                    elem.extend(new_elem)
            else:
                assert torch.is_tensor(value)
                old_dict.setdefault(key, []).extend(value)

    # indexes a list (inside of a list) inside of a dictionary
    @staticmethod
    def index_dict_list(dict_list, ind):
        list_dict = {}
        for k, v in dict_list.items():
            if not v or isinstance(v[0], list):
                new_v = [elem[ind].unsqueeze(0) for elem in v]
            else:
                new_v = v[ind].unsqueeze(0)
            list_dict[k] = new_v
        return list_dict

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

    def paths(self, new_exp_name=None):
        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)
        if new_exp_name:  # save a parallel version to work with
            directory = os.path.join(self.models_path, new_exp_name)
            ep = self.epoch
        else:
            directory = os.path.join(self.models_path, self.exp_name)
            ep = self.epoch
        if not os.path.exists(directory):
            os.mkdir(directory)
        paths = {'settings': os.path.join(directory, 'settings.json'),
                 'model': os.path.join(directory, f'model_epoch{ep}.pt'),
                 'optim': os.path.join(directory, f'optimizer_epoch{ep}.pt'),
                 'train_hist': os.path.join(directory, 'train_losses.json'),
                 'val_hist': os.path.join(directory, 'val_losses.json')}
        return paths


class VAETrainer(Trainer):
    clf = svm.SVC(kernel='linear')
    bin = 'pcdvae'  # minio bin
    saved_accuracies = {}

    def __init__(self, model, exp_name, block_args):
        self.acc = None
        self.cf = None
        self._loss = get_vae_loss(block_args)
        super().__init__(model, exp_name, **block_args)

        return

    def test(self, partition='val', m=2048):
        self.model.decode.m = m
        super().test(partition=partition)
        return

    def update_m_training(self, m):
        self.model.decode.m_training = m

    def clas_metric(self, final=False):
        # No rotation here
        self.train_loader.dataset.rotation = False
        self.test(partition='train')
        self.train_loader.dataset.rotation = True
        x_train = np.array([z.numpy() for z in self.test_outputs['z']])
        y_train = np.array([z.numpy() for z in self.test_targets])
        shuffle = np.random.permutation(y_train.shape[0])
        x_train = x_train[shuffle]
        y_train = y_train[shuffle]
        print('Fitting the classifier ...')
        self.clf.fit(x_train, y_train)
        partition = 'test' if final else 'val'
        self.test(partition=partition)
        x_test = np.array([z.numpy() for z in self.test_outputs['z']])
        y_test = np.array([z.numpy() for z in self.test_targets])
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
        z = torch.stack(self.test_outputs['z'])
        pca = PCA(3)
        z_np = z.numpy()
        z_red = pca.fit_transform(z_np)
        labels = torch.stack(self.test_targets).cpu().numpy()
        highlight_z = z_red[(highlight_label == labels)]
        pc_show([torch.FloatTensor(z_red), highlight_z], colors=['blue', 'red'])

    def loss(self, output, inputs, targets):
        return self._loss(output, inputs, targets)

def get_vae_trainer(model, exp_name, block_args):
    return VAETrainer(model, exp_name, block_args)
