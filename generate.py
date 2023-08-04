import torch

from src.options import parse_args_and_set_seed
from src.dataset import EmptyDataset
from src.model import get_model
from src.trainer import get_trainer
from src.viz_pc import infer_and_visualize


def generate_random_samples():
    args = parse_args_and_set_seed(task='gen_viz', description='Generate, visualize and render samples')
    assert args.model_head == 'VQVAE', 'Only VQVAE models can do random generation'
    model = get_model(**vars(args))
    datasets = dict(train_loader=EmptyDataset())
    trainer = get_trainer(model, datasets, args=args)
    trainer.load(args.load_checkpoint if args.load_checkpoint else None)

    z_bias = torch.zeros(args.gen, args.z_dim, device=args.device)
    z_bias[:, args.bias_dim] = args.bias_value
    infer_and_visualize(trainer.model, args, n_clouds=args.gen, mode='gen', z_bias=z_bias)



if __name__ == '__main__':
    generate_random_samples()
