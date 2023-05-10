import os
import torch
from src.options import parse_args_and_set_seed
from src.dataset import get_loaders, get_cw_loaders
from src.model import get_model
from src.trainer import get_trainer, get_cw_trainer


def train_second_encoding():
    args = parse_args_and_set_seed(description='Train second encoding')
    assert args.model_head == 'VQVAE', 'Only VQVAE supported'
    model = get_model(**vars(args))
    model.double_encoding = True
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    test_partition = 'train' if args.eval_train else 'test' if args.final else 'val'
    # TODO: load the model normally from trainer when fixing the architecture (careful with self.epoch)
    # trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    load_path = os.path.join('models', args.exp_name, f'model_epoch{args.epochs}.pt')
    assert os.path.exists(load_path), 'No pretrained experiment in ' + load_path
    model_state = torch.load(load_path, map_location=args.device)
    assert args.vae_load < 1, 'Only loading the last saved version is supported'
    for name in list(model_state):
        if args.vae_load == -1:
            model.cw_encoder.reset_parameters()
            if name[:10] == 'cw_encoder' or name[:10] == 'cw_decoder':
                model_state.popitem(name)  # temporary feature to experiment with different cw_encoders
    vqvae_trainer = get_trainer(model, loaders, args)
    vqvae_trainer.model.load_state_dict(model_state, strict=False)
    cw_train_loader, cw_test_loader = get_cw_loaders(vqvae_trainer, test_partition, args.vae_batch_size)
    cw_loaders = dict(train_loader=cw_train_loader, val_loader=None, test_loader=cw_test_loader)
    cw_trainer = get_cw_trainer(vqvae_trainer, cw_loaders, args)
    if args.eval:  # when final the test partition is made from the validation
        cw_trainer.test('test', all_metrics=True, denormalise=args.de_normalise, save_outputs=True)
        cw_trainer.show_latent()

    else:
        while args.vae_epochs > cw_trainer.epoch:
            cw_trainer.train(args.vae_checkpoint)
            cw_trainer.test(partition='test', all_metrics=True, denormalise=args.denormalise)
        cw_trainer.save()


if __name__ == '__main__':
    train_second_encoding()
