import os
import torch
from src.options import parse_args_and_set_seed
from src.dataset import get_loaders, get_cw_loaders
from src.model import get_model
from src.trainer import get_trainer, get_cw_trainer


def train_second_encoding():
    args = parse_args_and_set_seed(task='train_vae', description='Train second encoding')
    assert args.model_head == 'VQVAE', 'Only VQVAE supported'
    model = get_model(**vars(args))
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    test_partition = 'train' if args.eval_train else 'test' if args.final else 'val'

    # TODO: load the model normally from trainer when fixing the architecture (careful with self.epoch)
    # trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    load_path = os.path.join('models', args.exp_name, f'model_epoch{args.epochs}.pt')
    assert os.path.exists(load_path), 'No pretrained experiment in ' + load_path
    model_state = torch.load(load_path, map_location=args.device)
    if not args.vae_load:
        for name in list(model_state):
            if name[:10] == 'cw_encoder' or name[:10] == 'cw_decoder':
                model_state.popitem(name)  # temporary feature to experiment with different cw_encoders
    model.load_state_dict(model_state, strict=False)
    vqvae_trainer = get_trainer(model, loaders, args)
    vqvae_trainer.epoch = args.epochs
    cw_train_loader, cw_val_loader = get_cw_loaders(vqvae_trainer, 'train', test_partition, args.vae_batch_size)
    cw_loaders = dict(train_loader=cw_train_loader, val_loader=cw_val_loader, test_loader=None)
    cw_trainer = get_cw_trainer(vqvae_trainer, cw_loaders, args)

    while args.vae_epochs > cw_trainer.epoch:
        cw_trainer.train(args.vae_checkpoint)
        cw_trainer.test(partition=test_partition, all_metrics=True, de_normalize=args.de_normalize)
        cw_trainer.save()
    vqvae_trainer.test(partition=test_partition, all_metrics=True, de_normalize=args.de_normalize)
    cw_trainer.test(test_partition, all_metrics=True, de_normalize=args.de_normalize, save_outputs=True)
    cw_trainer.show_latent()


if __name__ == '__main__':
    train_second_encoding()
