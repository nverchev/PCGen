import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.options import parse_args_and_set_seed
from src.dataset import get_loaders
from src.model import get_model
from src.trainer import get_trainer


def train_model():
    args = parse_args_and_set_seed(description='Train a (loaded) model')
    model = get_model(**vars(args))
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer = get_trainer(model, loaders, args=args)
    test_partition = 'train' if args.eval_train else 'test' if args.final else 'val'

    if args.load:
        trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    else:
        # TearingNet starts from a pretrained FoldingNet model
        if args.decoder_name == 'TearingNet':
            exp_name_split = args.exp_name.split('_')
            exp_name_split[1] = 'FoldingNet'
            load_path = os.path.join('models', '_'.join(exp_name_split), f'model_epoch{args.epochs}.pt')
            assert os.path.exists(load_path), 'No pretrained FoldingNet experiment in ' + load_path
            state_dict = torch.load(load_path, map_location=args.device)
            trainer.model.load_state_dict(state_dict, strict=False)

    # Uncomment to plot learning curves
    while args.epochs > trainer.epoch:
        trainer.train(args.checkpoint)
        # if args.model_head == "VQVAE":
        #     trainer.test(partition='train', save_outputs=True)
        #     idx = trainer.test_outputs['cw_idx']
        #     idx = torch.stack(idx).sum(0)
        #     unused_idx = (idx == 0)
        #     for i in range(args.cw_dim // args.embedding_dim):
        #         p = np.array(idx[i])
        #         p = p / p.sum()
        #         for j in range(args.book_size):
        #             if unused_idx[i, j]:
        #                 k = np.random.choice(np.arange(args.book_size), p=p)
        #                 trainer.model.codebook.data[i, j] = trainer.model.codebook.data[i, k]
        if not args.final:
            trainer.test(partition='val')
        if args.training_plot:
            trainer.plot_loss_metric(start=trainer.epoch - 10 * args.checkpoint, loss_metric='Chamfer', update=True)
        trainer.save()
    trainer.test(partition=test_partition, all_metrics=True, de_normalize=args.de_normalize)


if __name__ == '__main__':
    train_model()
