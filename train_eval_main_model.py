import os
import torch
import numpy as np
from src.options import parse_process_args_and_set_seed
from src.trainer import get_trainer


def train_eval_model():
    args = parse_process_args_and_set_seed(task='train', description='Train or evaluate a model')
    trainer = get_trainer(args=args)
    if args.model_head == 'Oracle':
        return trainer.test(partition=args.test_partition, all_metrics=True, de_normalize=args.de_normalize)
    if args.load:
        trainer.load(args.load_checkpoint)
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
        if args.model_head == "VQVAE":
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
            start = max(trainer.epoch - 10 * args.checkpoint, 0)
            trainer.plot_learning_curves(start=start, loss_or_metric='Chamfer', title='PC Encoding')
        trainer.save()
    trainer.test(partition=args.test_partition, all_metrics=True, de_normalize=args.de_normalize)
    if args.training_plot:
        trainer.plot_learning_curves(loss_or_metric='Chamfer', title='PC Encoding')


if __name__ == '__main__':
    train_eval_model()
