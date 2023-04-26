import os
import torch
from src.options import parse_args_and_set_seed
from src.dataset import get_loaders
from src.model import get_model
from src.trainer import get_trainer, get_cw_trainer


def train_second_encoding():
    args = parse_args_and_set_seed(description='Train second encoding')
    assert args.model_head == 'VQVAE', 'Only VQVAE supported'
    model = get_model(**vars(args))
    model.double_encoding = True
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    cw_trainer = get_cw_trainer(model, loaders, args)
    if args.eval:  # when final the test partition is made from the validation
        cw_trainer.test('test', all_metrics=True, denormalise=args.denormalise, save_outputs=True)
        cw_trainer.show_latent()

    else:
        while args.vae_epochs > cw_trainer.epoch:
            cw_trainer.train(args.vae_checkpoint)
            cw_trainer.test(partition='test', all_metrics=True, denormalise=args.denormalise)
        cw_trainer.save()


if __name__ == '__main__':
    train_second_encoding()
