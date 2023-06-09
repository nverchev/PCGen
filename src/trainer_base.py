import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch import autocast
import os
import warnings
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from collections import UserDict, defaultdict
import sys
import typing


# Apply recursively lists or dictionaries until check
def apply(obj: typing.Any, check, f) -> typing.Any:  # changes device in dictionary and lists
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.info = {}

    def __getitem__(self, key_or_index: typing.Union[int, str]):
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
dict['Criterion'] = loss to backprop (summed over batch)
'''


class Trainer(metaclass=ABCMeta):
    quiet_mode = False  # less output
    max_stored_output = np.inf  # maximum amount of stored evaluated test samples

    def __init__(self, model, exp_name, device, optimizer, optim_args, train_loader, val_loader=None, test_loader=None,
                 model_pardir='./models', amp=False, scheduler=None, **block_args):
        self.epoch = 0
        self.device = device  # to cuda or not to cuda?
        self.model = model.to(device)
        self.exp_name = exp_name  # name used for saving and loading
        self.scheduler = scheduler
        self.settings = {**model.settings, **block_args}
        self.optimizer_settings = optim_args.copy()  # may be overwritten by scheduling
        self.optimizer = optimizer(**self.optimizer_settings)
        self.scaler = GradScaler(enabled=amp and self.device.type == 'cuda')
        self.amp = amp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_log, self.val_log = defaultdict(dict), defaultdict(dict)
        self.test_metadata, self.test_outputs = None, None  # store last test evaluation
        self.saved_test_metrics = {}  # saves metrics of last evaluation
        self.model_pardir = model_pardir
        json.dump(self.settings, open(self.paths()['settings'], 'w'), default=vars, indent=4)

    @property
    def optimizer_settings(self):  # settings shown depend on epoch
        if self.scheduler is None:
            return {'params': self._optimizer_settings[0],
                    **self._optimizer_settings[1]}
        else:  # the scheduler modifies the learning rate(s)
            init_learning = self._optimizer_settings[0]
            scheduled_learning = []
            for group in init_learning:
                scheduled_learning.append({
                    'params': group['params'],
                    'lr': self.scheduler(group['lr'], self.epoch)
                })
            return {'params': scheduled_learning,
                    **self._optimizer_settings[1]}

    @optimizer_settings.setter
    def optimizer_settings(self, optim_args):
        lr = optim_args.pop('lr')
        if isinstance(lr, dict):  # support individual lr for each parameter (for fine-tuning for example)
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
            print('Experiment name: ', self.exp_name)
        for _ in range(num_epoch):
            self.update_learning_rate(self.optimizer_settings['params'])
            self.epoch += 1
            if self.quiet_mode:
                print('\r====> Epoch:{:4d}'.format(self.epoch), end='')
            else:
                print('====> Epoch:{:4d}'.format(self.epoch))
            self._hook_before_training_epoch()
            self.model.train()
            self._run_session(partition='train')
            self._hook_after_training_epoch()
            if self.val_loader and val_after_train:  # check losses on val
                self.model.eval()
                with torch.inference_mode():
                    self._run_session(partition='val')
        return

    @torch.inference_mode()
    def test(self, partition, save_outputs=False, **kwargs):  # runs and stores evaluated test samples
        if not self.quiet_mode:
            print('Version ', self.exp_name)
        self.model.eval()
        self._run_session(partition=partition, save_outputs=save_outputs)
        return

    def _run_session(self, partition='train', save_outputs=False):
        if partition == 'train':
            loader = self.train_loader
            dict_log = self.train_log[str(self.epoch)]
        elif partition == 'val':
            loader = self.val_loader
            dict_log = self.val_log[str(self.epoch)]
        elif partition == 'test':
            loader = self.test_loader
            dict_log = self.saved_test_metrics
        else:
            raise ValueError('partition options are: "train", "val", "test" ')
        if save_outputs:
            self.test_metadata, self.test_outputs = TorchDictList(), TorchDictList()
            self.test_metadata.info = dict(partition=partition, max_ouputs=self.max_stored_output)

        epoch_log = defaultdict(float)
        num_batch = len(loader)
        with tqdm(enumerate(loader), total=num_batch, disable=self.quiet_mode, file=sys.stderr) as tqdm_loader:
            epoch_seen = 0
            for batch_idx, (inputs, targets, indices) in tqdm_loader:
                epoch_seen += indices.shape[0]
                inputs, targets = self.recursive_to([inputs, targets], self.device)
                inputs_aux = self.helper_inputs(inputs, targets)
                with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    outputs = self.model(**inputs_aux)
                    if torch.is_inference_mode_enabled():
                        batch_log = self.metrics(outputs, inputs, targets)
                    else:
                        batch_log = self.loss(outputs, inputs, targets)
                        criterion = batch_log['Criterion']
                    for loss_metric, value in batch_log.items():
                        epoch_log[loss_metric] += value.item()
                if not torch.is_inference_mode_enabled():
                    if torch.isnan(criterion):
                        raise ValueError('Criterion is nan')
                    if torch.isinf(criterion):
                        raise ValueError('Criterion is inf')
                    self.scaler.scale(criterion).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if not self.quiet_mode:
                        if batch_idx % (num_batch // 10 or 1) == 0:
                            tqdm_loader.set_postfix({'Seen': epoch_seen,
                                                     'Loss': criterion.item()})
                # if you get memory error, limit max_stored_output
                if save_outputs and self.max_stored_output > (batch_idx + 1) * loader.batch_size:
                    self.test_outputs.extend_dict(self.recursive_to(outputs, 'detach_cpu'))
                    self.test_metadata.extend_dict(dict(indices=indices))
                    self.test_metadata.extend_dict(dict(targets=targets.cpu()))

        print('Average {} {} :'.format(partition, 'metrics' if torch.is_inference_mode_enabled() else 'losses'))
        for loss_metric, value in epoch_log.items():
            value = value / num_batch if loss_metric == 'Criterion' else value / epoch_seen
            if torch.is_inference_mode_enabled() ^ (partition == 'train'):  # Does not overwrite training curve
                dict_log[loss_metric] = value
            if not self.quiet_mode:
                print('{}: {:.4e}'.format(loss_metric, value), end='\t')
        print()
        return

    @abstractmethod
    def loss(self, output, inputs, targets):
        pass

    def metrics(self, output, inputs, targets):
        return self.loss(output, inputs, targets)

    def helper_inputs(self, inputs, labels):
        return {'x': inputs}

    def plot_loss_metric(self, plot_train=True, plot_val=True, loss_metric='Criterion', start=0, update=False):
        plt.cla()
        ax = plt.gca()
        if self.train_log and plot_train:
            epoch_keys = [epoch for epoch in self.train_log.keys() if int(epoch) >= start]
            epochs = [int(epoch) for epoch in epoch_keys]
            values = [self.train_log[epoch][loss_metric] for epoch in epoch_keys]
            ax.plot(epochs, values, label='train', color='blue')
        if self.val_log and plot_val:
            epoch_keys = [epoch for epoch in self.val_log.keys() if int(epoch) >= start]
            epochs = [int(epoch) for epoch in epoch_keys]
            values = [self.val_log[epoch][loss_metric] for epoch in epoch_keys]
            ax.plot(epochs, values, label='val', color='green')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(loss_metric)
        ax.set_title(f'{self.exp_name}')
        ax.legend()
        if update:
            plt.pause(0.1)
        else:
            plt.show()
        return

    # Change device recursively to tensors inside a list or a dictionary
    @staticmethod
    def recursive_to(obj, device):  # changes device in dictionary and lists
        if device == 'detach_cpu':
            return apply(obj, check=torch.is_tensor, f=lambda x: x.detach().cpu())
        return apply(obj, check=torch.is_tensor, f=lambda x: x.to(device))

    def save(self, new_exp_name=''):
        self.model.eval()
        paths = self.paths(new_exp_name)
        torch.save(self.model.state_dict(), paths['model'])
        torch.save(self.optimizer.state_dict(), paths['optim'])
        for json_file_name in ['train_log', 'val_log', 'saved_test_metrics'] + (['settings'] if new_exp_name else []):
            json.dump(self.__getattribute__(json_file_name), open(paths[json_file_name], 'w'))
        print('Model saved at: ', paths['model'])
        return

    def load(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        else:
            past_epochs = []  # here it looks for the most recent model
            local_path = os.path.join(self.model_pardir, self.exp_name)
            if os.path.exists(local_path):
                for file in os.listdir(local_path):
                    if file[:5] == 'model':
                        past_epochs.append(int(''.join(filter(str.isdigit, file))))
            if not past_epochs:
                warnings.warn('No saved models found. Training from scratch.', UserWarning)
                return
            else:
                self.epoch = max(past_epochs)
        paths = self.paths()

        # TODO: remove strict flag
        self.model.load_state_dict(torch.load(paths['model'], map_location=torch.device(self.device)))
        # self.optimizer.load_state_dict(torch.load(paths['optim'], map_location=torch.device(self.device)))
        for json_file_name in ['train_log', 'val_log', 'saved_test_metrics']:
            json_file = json.load(open(paths[json_file_name]))
            # Stop from loading logs of future epochs
            if json_file_name in ['train_log', 'val_log']:
                for epoch in list(json_file):
                    if int(epoch) > self.epoch:
                        json_file.pop(epoch)
            self.__setattr__(json_file_name, defaultdict(dict, json_file))
        print('Loaded: ', paths['model'])
        return

    def paths(self, new_exp_name=None, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        if not os.path.exists(self.model_pardir):
            os.mkdir(self.model_pardir)
        if new_exp_name:  # save a parallel version to work with
            directory = os.path.join(self.model_pardir, new_exp_name)
        else:
            directory = os.path.join(self.model_pardir, self.exp_name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        paths = {json_file: os.path.join(directory, f'{json_file}.json') for json_file in
                 ['settings', 'train_log', 'val_log', 'saved_test_metrics']}
        paths.update({pt_file: os.path.join(directory, f'{pt_file}_epoch{epoch}.pt') for pt_file in ['model', 'optim']})
        return paths

    def _hook_before_training_epoch(self):
        pass

    def _hook_after_training_epoch(self):
        pass
