import numpy as np
import torch.optim as optim


class NoSchedule:

    def __call__(self, base_lr, epoch):
        return base_lr

    def __str__(self):
        return 'NoSchedule'


class ExponentialSchedule:
    def __init__(self, exp_decay=.975):
        self.exp_decay = exp_decay

    def __call__(self, base_lr, epoch):
        return base_lr * self.exp_decay ** epoch

    def __str__(self):
        return 'ExponentialSchedule'


class CosineSchedule:
    def __init__(self, decay_steps=60, min_decay=0.1):
        self.decay_steps = decay_steps
        self.min_decay = min_decay

    def __call__(self, base_lr, epoch):
        min_lr = self.min_decay * base_lr
        if epoch > self.decay_steps:
            return min_lr
        return min_lr + (base_lr - min_lr) * (1 + np.cos(np.pi * epoch / self.decay_steps) / 2)

    def __str__(self):
        return 'CosineSchedule'


def get_opt(opt, initial_learning_rate, weight_decay=0):
    optimizer = {'Adam': optim.Adam,
                 'AdamW': optim.AdamW,
                 'SGD': optim.SGD,
                 'SGD_momentum': optim.SGD
                 }

    optimi_args = {'Adam': {'weight_decay': weight_decay, 'lr': initial_learning_rate},
                   'AdamW': {'weight_decay': weight_decay, 'lr': initial_learning_rate},
                   'SGD': {'weight_decay': weight_decay, 'lr': initial_learning_rate},
                   'SGD_momentum': {'weight_decay': weight_decay, 'lr': initial_learning_rate,
                                    'momentum': 0.9, 'nesterov': True}}
    return optimizer[opt], optimi_args[opt]
