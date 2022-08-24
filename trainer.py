import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import re
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import svm, metrics
import torch.cuda.amp as amp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from losses import get_vae_loss, get_classification_loss
from plot_PC import pc_show

'''
This abstract class manages training and general utilities.
It works together with a class defining the loss.
This loss returns a dictionary dict with 
dict["Criterion"] = loss to backprop

To save and load on a separate sever, it expects a Minio object from the minio library.
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

    def __init__(self, model, exp_name, device, optim, train_loader, val_loader=None,
                 test_loader=None, minioClient=None, dir_path='./', **block_args):

        torch.manual_seed = 112358
        self.epoch = 0
        self.device = device  # to cuda or not to cuda?
        self.model = model.to(device)  # model is not copied
        self.exp_name = exp_name  # name used for saving and loading
        self.schedule = block_args['schedule']
        self.settings = {**model.settings, **block_args}
        self.optimizer_settings = block_args['optim_args'].copy()
        self.optimizer = optim(**self.optimizer_settings)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_losses = {loss: [] for loss in self.losses}
        self.val_losses = {loss: [] for loss in self.losses}
        self.test_losses = {loss: [] for loss in self.losses}
        self.test_targets, self.test_outputs = [], {}
        self.converge = 1  # if 0 kills session
        self.minio = minioClient
        self.dir_path = dir_path
        self.minio_path = staticmethod(lambda path: path[len(dir_path):]).__func__  # removes dir path
        self.test_targets, self.test_outputs = [], {}  # stored in RAM
        settings_path = self.paths()['settings']
        json.dump(self.settings, open(settings_path, 'w'), default=vars)

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
                print('\r====> Epoch:{:3d}'.format(self.epoch), end="")
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

        epoch_loss = {loss: 0 for loss in self.losses}
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
            for loss in self.losses:
                epoch_loss[loss] += batch_loss[loss].item()
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
                if batch_idx % (len(loader) // 10 or 1) == 0:
                    iterable.set_postfix({'Seen': batch_idx * loader.batch_size,
                                          'Loss': criterion.item()})
                if batch_idx == len(loader) - 1:  # clear after last
                    iterable.set_description('')

        for loss in self.losses:
            epoch_loss[loss] /= num_batch
            if not save_outputs:  # do not save history when testing
                dict_losses[loss].append(epoch_loss[loss])
        if not self.quiet_mode:
            print('Average {} losses :'.format(partition))
            for loss in self.losses:
                print('{}: {:.4f}'.format(loss, epoch_loss[loss]), end='\t')
            print()
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
        plt.title(f"{self.exp_name}")
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

    def save(self, new_exp_name=None):
        self.model.eval()
        paths = self.paths(new_exp_name)
        torch.save(self.model.state_dict(), paths['model'])
        torch.save(self.optimizer.state_dict(), paths['optim'])
        json.dump(self.train_losses, open(paths['train_hist'], 'w'))
        json.dump(self.val_losses, open(paths['val_hist'], 'w'))
        if self.minio is not None:
            for file in paths.values():
                self.minio.fput_object(self.bin, self.minio_path(file), file)
        print("Model saved at: ", paths['model'])
        return

    def load(self, epoch=None):
        directory = self.exp_name

        if epoch is not None:
            self.epoch = epoch
        else:
            past_epochs = []  # here it looks for the most recent model
            if self.minio is not None:
                for file in self.minio.list_objects(self.bin, recursive=True):
                    file_dir, *file_name = file.object_name.split("/")
                    if file_dir == directory and file_name[0][:5] == 'model':
                        past_epochs.append(int(re.sub("\D", "", file_name[0])))
            local_path = os.path.join(self.dir_path, self.exp_name)
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
                if file[-13:] != 'settings.json':
                    self.minio.fget_object(self.bin, self.minio_path(file), file)
        self.model.load_state_dict(torch.load(paths['model'],
                                              map_location=torch.device(self.device)))
        self.optimizer.load_state_dict(torch.load(paths['optim'],
                                                  map_location=torch.device(self.device)))
        self.train_losses = json.load(open(paths['train_hist']))
        self.val_losses = json.load(open(paths['val_hist']))
        print("Loaded: ", paths['model'])
        return

    def paths(self, new_exp_name=None):
        if new_exp_name:  # save a parallel version to work with
            directory = os.path.join(self.dir_path, new_exp_name)
            ep = self.epoch
        else:
            directory = os.path.join(self.dir_path, self.exp_name)
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

    def __init__(self, model, recon_loss, exp_name, block_args):
        self.acc = None
        self.cf = None
        self._loss = get_vae_loss(recon_loss)
        self.losses = self._loss.losses  # losses must be defined before super().__init__()
        model.settings.update({'c_KLD': self._loss.c_KLD})
        super().__init__(model, exp_name, **block_args)

        return

    def loss(self, output, inputs, targets):
        return self._loss(output, inputs, targets)

    def test(self, partition='val', m=2048):
        self.model.decode.m = m
        super().test(partition=partition)
        return

    def update_m_training(self, m):
        self.model.decode.m_training = m

    def clas_metric(self, final=False):
        # No rotation here
        self.train_loader.dataset.rotation = False
        self.test(partition="train")
        self.train_loader.dataset.rotation = True
        x_train = np.array([z.numpy() for z in self.test_outputs['z']])
        y_train = np.array([z.numpy() for z in self.test_targets])
        shuffle = np.random.permutation(y_train.shape[0])
        x_train = x_train[shuffle]
        y_train = y_train[shuffle]
        print("Fitting the classifier ...")
        self.clf.fit(x_train, y_train)
        partition = "test" if final else "val"
        self.test(partition=partition)
        x_test = np.array([z.numpy() for z in self.test_outputs['z']])
        y_test = np.array([z.numpy() for z in self.test_targets])
        y_hat = self.clf.predict(x_test)
        self.acc = (y_hat == y_test).sum() / y_hat.shape[0]
        print("Accuracy: ", self.acc)
        self.cf = metrics.confusion_matrix(y_hat, y_test, normalize='true')
        print('Mean Accuracy;', np.diag(self.cf).astype(float).mean())
        directory = os.path.join(self.dir_path, self.exp_name)
        accuracy_path = os.path.join(directory, "svm_accuracies.json")
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


class ClassificationTrainer(Trainer):
    saved_metrics = {}
    _metrics = {}
    average = "macro"
    bin = 'pcdvae'  # minio bin

    def __init__(self, model, loss, exp_name, block_args):
        self._loss = get_classification_loss(loss)(model.num_classes)
        self.losses = self._loss.losses  # losses must be defined before super().__init__()
        super().__init__(model, exp_name, **block_args)
        self.wrong_indices = None
        self.targets = None
        self.test_pred = None
        self.test_probs = None
        return

    def loss(self, output, inputs, targets):
        return self._loss(output, inputs, targets)

    # overwrites Trainer method
    def test(self, partition='test'):
        super().test(partition=partition)  # stored in RAM
        y = torch.stack(self.test_outputs['y'])
        self.test_probs = F.softmax(y, dim=-1)
        self.test_pred = torch.argmax(self.test_probs, dim=1)
        self.targets = torch.stack(self.test_targets)
        right_pred = (self.test_pred == self.targets)
        self.wrong_indices = torch.nonzero(~right_pred).squeeze()
        self.calculate_metrics()
        directory = os.path.join(self.dir_path, self.exp_name)
        metrics_path = os.path.join(directory, "metrics.json")
        self.saved_metrics[self.epoch] = self._metrics.copy()
        json.dump(self.saved_metrics, open(metrics_path, 'w'))
        return

    @property
    def metrics(self):
        self.test(partition='val')
        return self._metrics

    def calculate_metrics(self):
        avg_type = self.average.capitalize() + ' ' if self.test_probs.size(1) > 1 else ""
        # calculates common and also gives back the indices of the wrong guesses

        self._metrics['Accuracy'] = metrics.accuracy_score(self.targets, self.test_pred)
        one_hot_targets = torch.zeros(len(self.targets), max(self.targets) + 1).scatter(1, self.targets.view(-1, 1), 1)
        one_hot_pred = torch.zeros_like(one_hot_targets).scatter(1, self.test_pred.view(-1, 1), 1)
        correct = (one_hot_pred * one_hot_targets)
        self._metrics['Mean Accuracy'] = (correct.sum(0) / one_hot_targets.sum(0)).mean().item()
        self._metrics[avg_type + 'F1 Score'] = metrics.f1_score(self.targets, self.test_pred, average=self.average)
        self._metrics[avg_type + 'Jaccard Score'] = metrics.jaccard_score(self.targets,
                                                                          self.test_pred, average=self.average)
        self._metrics[avg_type + 'AUC ROC'] = \
            metrics.roc_auc_score(self.targets, self.test_probs,
                                  average=self.average, multi_class='ovr')
        for metric, value in self._metrics.items():
            print(metric + f': {value:.4f}', end='\t')
        print('')
        return


def get_class_trainer(model, loss, exp_name, block_args):
    return ClassificationTrainer(model, loss, exp_name, block_args)


def get_vae_trainer(model, recon_loss, exp_name, block_args):
    return VAETrainer(model, recon_loss, exp_name, block_args)

# class VAEMetric():
#     # overwrites Trainer method
#     def test(self, on='val', batch_test=64):
#         super().test(on=on)
#         if on == 'val':
#             inputs = self.val_loader.dataset.dataset.pcd
#         elif on == 'test':
#             inputs = self.test_loader.dataset.dataset.pcd
#
#         l_test = len(inputs)
#         recons = self.test_outputs['recon']
#         n = inputs[0].size()[1]
#         m = recons[0].size()[1]
#         sigma6 = torch.tensor(0.01)
#         mu = torch.vstack(self.test_outputs['mu'])
#         logvar = torch.vstack(self.test_outputs['log_var'])
#         if len(recons[0].size()) == 4:
#             n_samples = recons.size()[1]
#             inputs = inputs.unsqueeze(1).expand(-1, n_samples, -1, -1)
#         KLD, _ = kld_loss(mu, logvar)
#         nll_recon = 0
#         chamfer_recon = 0
#         for i in range(0, l_test, batch_test):
#             batch_inputs = torch.vstack(inputs[i:i + batch_test])
#             batch_recons = torch.vstack(recons[i:i + batch_test])
#             pairwise_dist = square_distance(batch_inputs, batch_recons)
#             chamfer_recon += chamfer(pairwise_dist)
#             nll_recon += nll(pairwise_dist, sigma6, n, m)
#
#         print(f'KLD: {KLD:.4f}', end='\t')
#         print(f'Chamfer: {chamfer_recon / l_test:.4f}', end='\t')
#         print(f'NLL: {nll_recon / l_test:.4f}', end='\t')
