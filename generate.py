import torch

from src.options import parse_process_args_and_set_seed
from src.model import get_model
from custom_trainer import Trainer
from src.viz_pc import infer_and_visualize


def generate_random_samples():
    args = parse_process_args_and_set_seed(task='gen_viz', description='Generate, visualize and render samples')
    assert args.model_head == 'VQVAE', 'Only VQVAE models can do random generation'
    model = get_model(**vars(args)).to(args.device)
    model_path = Trainer.paths(args.exp_name, args.epochs)['model']
    model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
    z_bias = torch.zeros(args.gen, args.z_dim, device=args.device)
    z_bias[:, args.bias_dim] = args.bias_value
    infer_and_visualize(model, args, n_clouds=args.gen, mode='gen', z_bias=z_bias)


if __name__ == '__main__':
    generate_random_samples()
