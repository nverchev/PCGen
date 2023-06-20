import os
import numpy as np
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
    trainer = get_trainer(model, loaders, args=args)
    if args.load:
        trainer.load(args.load_checkpoint if args.load_checkpoint else None)
        trainer.model.cw_encoder.recursive_reset_parameters()
    trainer.test('val')


    # Uncomment to plot learning curves
    while args.epochs > trainer.epoch:
        trainer.train(args.checkpoint)
        if args.model_head == "VQVAE" and trainer.epoch != 0:
            trainer.test(partition='train', save_outputs=True)
            idx = trainer.test_outputs['one_hot_idx']
            idx = torch.stack(idx).sum(0)
            unused_idx = (idx == 0)
            for i in range(args.w_dim // args.embedding_dim):
                p = np.array(idx[i])
                p = p / p.sum()
                for j in range(args.book_size):
                    if unused_idx[i, j]:
                        k = np.random.choice(np.arange(args.book_size), p=p)
                        used_embedding = trainer.model.codebook.data[i, k]
                        noise = args.vq_noise * torch.randn_like(used_embedding)
                        trainer.model.codebook.data[i, j] = used_embedding + noise
        if not args.final:
            trainer.test(partition='val')
        if args.training_plot:
            trainer.plot_loss_metric(start=trainer.epoch - 10 * args.checkpoint, loss_metric='Chamfer', update=True)
        trainer.save()
    trainer.test(partition=test_partition, all_metrics=True, de_normalize=args.de_normalize)
    if args.training_plot:
        trainer.plot_loss_metric(start=args.checkpoint, loss_metric='Chamfer')
if __name__ == '__main__':
    train_second_encoding()
