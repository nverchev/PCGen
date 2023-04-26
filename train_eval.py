import os
import torch
from src.options import parse_args_and_set_seed
from src.dataset import get_loaders
from src.model import get_model
from src.trainer import get_trainer


def train_eval():
    args = parse_args_and_set_seed(description='Train or eval a (loaded) model')
    model = get_model(**vars(args))
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    loaders = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer = get_trainer(model, loaders, args=args)
    test_partition = 'test' if args.final else 'val'
    if args.load == 0:
        trainer.load()
    elif args.load > 0:
        trainer.load(args.load)
    if args.eval:
        trainer.test(partition=test_partition, all_metrics=True, denormalise=args.denormalise)
    else:
        # TearingNet starts from a pretrained FoldingNet model
        if args.load == -1 and args.decoder_name == 'TearingNet':
            exp_name_split = args.exp_name.split('_')
            exp_name_split[1] = 'FoldingNet'
            load_path = os.path.join('models', '_'.join(exp_name_split), f'model_epoch{args.epochs}.pt')
            assert os.path.exists(load_path), 'No pretrained FoldingNet experiment in ' + load_path
            state_dict = torch.load(load_path, map_location=args.device)
            trainer.model.load_state_dict(state_dict, strict=False)

        while args.epochs > trainer.epoch:
            # if ae == "VQVAE" and trainer.epoch % 100 == 0 and trainer.epoch != 0:
            #     trainer.test(partition='train', m=m, save_outputs=True)
            #     idx = trainer.test_outputs['cw_idx']
            #     idx = torch.stack(idx).sum(0)
            #     unused_idx = (idx == 0)
            #     for i in range(cw_dim // dim_embedding):
            #         p = numpy.array(idx[i])
            #         p = p / p.sum()
            #         for j in range(book_size):
            #             if unused_idx[i, j]:
            #                 k = np.random.choice(np.arange(book_size), p=p)
            #                 trainer.model.codebook.data[i, j] = trainer.model.codebook.data[i, k]

            trainer.train(args.checkpoint)
            trainer.save()
            trainer.test(partition=test_partition)


if __name__ == '__main__':
    train_eval()
