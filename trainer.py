import numpy as np
import torch
import os
import json
import re
import torch.cuda.amp as amp
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from sklearn import svm
from dist_losses import get_loss

# @title Trainer
'''
This abstract class manages training and general utilites.
It works together with a class defining the loss.
This loss returns a dictionary dict with 
dict["Criterion"] = *list of losses to backprop*

To save and load, it expects a Minio object from the minio library.
This object downloads and uploads the model to a separate storage.
In order to get full access to this class utilities,
set up MinIO on your storage device (https://min.io)
install the minio api and then add the following:

from minio import Minio
minioClient = Minio(*Your storage name*,
                  access_key= *Your access key*,
                  secret_key= *Your secret key*,
                  secure=True)

'''


class Trainer(metaclass=ABCMeta):
    losses = []  # defined later with the loss function
    quiet_mode = False  # less output
    max_output = np.inf  # maximum amount of stored evaluated test samples
    bin = 'pcdvae'  # minio bin

    def __init__(self, model, version, device, optim, train_loader, val_loader=None,
                 test_loader=None, minioClient=None, dirpath='./', mp=False, **block_args):

        torch.manual_seed = 112358
        self.epoch = 0
        self.device = device  # to cuda or not to cuda?
        self.model = model.to(device)  # model is not copied
        self.version = version  # name used for saving and loading
        self.schedule = block_args['schedule']
        self.settings = {**model.settings, **block_args, 'Optimizer': str(optim)}
        self.optimizer_settings = block_args['optim_args'].copy()
        self.optimizer = optim(**self.optimizer_settings)
        self.mp = mp and device.type == 'cuda'  # mixed precision casting
        self.scaler = amp.GradScaler(enabled=mp)  # mixed precision backpropa
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_losses = {loss: [] for loss in self.losses}
        self.val_losses = {loss: [] for loss in self.losses}
        self.test_losses = {loss: [] for loss in self.losses}
        self.test_targets, self.test_outputs = [], {}
        self.converge = 1  # if 0 kills session
        self.minio = minioClient
        self.dirpath = dirpath
        self.miniopath = staticmethod(lambda path: path[len(dirpath):])  # removes dirpath

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
        print('Version ', self.version)
        for _ in range(num_epoch):
            self.update_learning_rate(self.optimizer_settings['params'])
            self.epoch += 1
            if self.quiet_mode:
                print('\r====> Epoch:{}'.format(self.epoch), end="")
            else:
                print('====> Epoch:{}'.format(self.epoch))
            self._run_session(mode='train')
            if self.val_loader and val_after_train:  # check losses on val
                self._run_session(mode='val', inference=True)  # best to test instead
        return

    def test(self, on='val'):  # runs and stores evaluated test samples
        print('Version ', self.version)
        self.test_targets, self.test_outputs = \
            self._run_session(mode=on, inference=True, save_outputs=True)  # stored in RAM
        return

    def _run_session(self, mode='train', inference=False,
                     save_outputs=False, max_output=None):
        if inference:
            self.model.eval()
            torch.set_grad_enabled(False)
        else:
            self.model.train()
            torch.set_grad_enabled(True)
        if mode == 'train':
            loader = self.train_loader
            dict_losses = self.train_losses
        elif mode == 'val':
            loader = self.val_loader
            dict_losses = self.val_losses
        elif mode == 'test':
            loader = self.test_loader
            dict_losses = self.test_losses
        else:
            raise ValueError('mode options are: "train", "val", "test" ')
        if save_outputs:
            test_targets, test_outputs = [], {}

        len_sess = len(loader.dataset)
        epoch_loss = {loss: 0 for loss in self.losses}
        num_batch = len(loader)
        iterable = enumerate(loader) if self.quiet_mode else \
            tqdm(enumerate(loader), total=num_batch)
        for batch_idx, (inputs, targets) in iterable:
            if self.converge == 0:
                return
            inputs, targets = self.to_recursive([inputs, targets], self.device)
            inputs_aux = self.helper_inputs(inputs, targets)
            with amp.autocast(self.mp):
                outputs = self.model(**inputs_aux)
                batch_loss = self.loss(outputs, inputs, targets)
            criterion = batch_loss['Criterion']
            for loss in self.losses:
                epoch_loss[loss] += batch_loss[loss].item()
            if not inference:
                if torch.isinf(criterion) or torch.isnan(criterion):
                    self.converge = 0
                self.scaler.scale(criterion).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            if save_outputs and \
                    self.max_output > (batch_idx + 1) * loader.batch_size:
                self.extend_dict_list(
                    test_outputs, self.to_recursive(outputs, 'detach_cpu'))
                test_targets.extend(self.to_recursive(targets, 'detach_cpu'))
            if not self.quiet_mode and mode == 'train':
                if batch_idx % (len(loader) // 10 or 1) == 0:
                    iterable.set_description(
                        'Train [{:4d}/{:4d} ]\tLoss {:4f}'.format(
                            batch_idx * loader.batch_size,
                            len_sess,
                            criterion.item())
                    )
                if batch_idx == len(loader) - 1:  # clear after last
                    iterable.set_description('')

        for loss in self.losses:
            epoch_loss[loss] /= num_batch
            if not save_outputs:  # do not save history when testing
                dict_losses[loss].append(epoch_loss[loss])
        if not self.quiet_mode:
            print('Average {} losses :'.format(mode))
            for loss in self.losses:
                print('{}: {:.4f}'.format(loss, epoch_loss[loss]), end='\t')
            print()
        if save_outputs:
            return test_targets, test_outputs
        else:
            return

    def helper_inputs(self, inputs, labels):
        return {'x': inputs}

    def plot_losses(self, loss):
        tidy_loss = " ".join([s.capitalize() for s in loss.split('_')])
        epochs = np.arange(self.epoch)
        plt.plot(epochs, self.train_losses[loss], label='train')
        if self.val_loader:
            plt.plot(epochs, self.val_losses[loss], label='val')
        plt.xlabel('Epochs')
        plt.ylabel(tidy_loss)
        plt.title(f"{self.version}")
        plt.show()
        return

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

    @staticmethod  # extends lists in dictionaries
    def extend_dict_list(old_dict, new_dict):
        if old_dict == {}:
            # struct is dict of lists / lists of lists of tensors
            old_dict.update({key: ([[]] if isinstance(value, list) else []) \
                             for key, value in new_dict.items()})

        for key, value in new_dict.items():
            if isinstance(value, list):
                for elem, new_elem in zip(old_dict[key], value):
                    assert torch.is_tensor(new_elem)
                    elem.extend(new_elem)
            else:
                assert torch.is_tensor(value)
                old_dict[key].extend(value)

    @staticmethod  # indexes a list (inside of a list) inside of a dictionary
    def index_dict_list(dict_list, ind):
        list_dict = {}
        for k, v in dict_list.items():
            if len(v) == 0 or isinstance(v[0], list):
                new_v = []
                for elem in v:
                    new_v.append(elem[ind].unsqueeze(0))
            else:
                new_v = v[ind].unsqueeze(0)
            list_dict[k] = new_v
        return list_dict

    @abstractmethod
    def loss(self, output, inputs, targets):
        pass

    def save(self, new_version=None):
        self.model.eval()
        paths = self.paths(new_version)
        torch.save(self.model.state_dict(), paths['model'])
        torch.save(self.optimizer.state_dict(), paths['optim'])
        json.dump(self.train_losses, open(paths['train_hist'], 'w'))
        json.dump(self.val_losses, open(paths['val_hist'], 'w'))
        if self.minio is not None:
            for file in paths.values():
                self.minio.fput_object(self.bin, self.miniopath(file), file)
        print("Model saved at: ", paths['model'])
        return

    def load(self, epoch=None):
        directory = self.version

        if epoch is not None:
            self.epoch = epoch
        else:
            past_epochs = []  # here it looks for the most recent model
            if self.minio is not None:
                for file in self.minio.list_objects(self.bin, recursive=True):
                    file_dir, *file_name = file.object_name.split("/")
                    if file_dir == directory and file_name[0][:5] == 'model':
                        past_epochs.append(int(re.sub("\D", "", file_name[0])))
            local_path = os.path.join(self.dirpath, self.version)
            if os.path.exists(local_path):
                for file in os.listdir(local_path):
                    if file[:5] == 'model':
                        past_epochs.append(int(re.sub("\D", "", file)))
            if len(past_epochs) == 0:
                print("No saved models found")
                return
            else:
                self.epoch = max(past_epochs)
        paths = self.paths()
        if self.minio is not None:
            for file in paths.values():
                self.minio.fget_object(self.bin, self.miniopath(file), file)
        self.model.load_state_dict(torch.load(paths['model'],
                                              map_location=torch.device(self.device)))
        self.optimizer.load_state_dict(torch.load(paths['optim'],
                                                  map_location=torch.device(self.device)))
        self.train_losses = json.load(open(paths['train_hist']))
        self.val_losses = json.load(open(paths['val_hist']))
        print("Loaded: ", paths['model'])
        return

    def paths(self, new_version=None):
        if new_version:  # save a parallel version to work with
            directory = os.path.join(self.dirpath, new_version)
            ep = self.epoch
        else:
            directory = os.path.join(self.dirpath, self.version)
            ep = self.epoch
        if not os.path.exists(directory):
            os.mkdir(directory)
        paths = {'model': os.path.join(directory, f'model_epoch{ep}.pt'),
                 'optim': os.path.join(directory, f'optimizer_epoch{ep}.pt'),
                 'train_hist': os.path.join(directory, 'train_losses.json'),
                 'val_hist': os.path.join(directory, 'val_losses.json')}
        return paths


class VAETrainer(Trainer):
    clf = svm.SVC()

    def __init__(self, model, version, block_args):
        super().__init__(model, version, **block_args)
        return

    def test(self, on='val', m=128):
        self.model.decode.m = m
        super().test(on=on)
        return

    def clas_metric(self):
        self.test(on="train")
        x_train = np.array([z.numpy() for z in self.test_outputs['z']])
        y_train = np.array([z.numpy() for z in self.test_targets])
        shuffle = np.random.permutation(y_train.shape[0])
        x_train = x_train[shuffle]
        y_train = y_train[shuffle]
        print("Fitting the classifier ...")
        self.clf.fit(x_train, y_train)
        self.test(on="val")
        x_val = np.array([z.numpy() for z in self.test_outputs['z']])
        y_val = np.array([z.numpy() for z in self.test_targets])
        y_hat = self.clf.predict(x_val)
        print("Accuracy: ", self.acc)
        return (y_hat == y_val).sum() / y_hat.shape[0]


def get_trainer(model, trainer_name, recon_loss, block_args):
    Loss = get_loss(recon_loss)

    class FinalTrainer(Loss, VAETrainer):
        pass

    return FinalTrainer(model, trainer_name, block_args)
