from src.options import parse_args_and_set_seed
from src.dataset import EmptyDataset
from src.model import get_model
from src.trainer import get_trainer
from src.viz_pc import show_pc, render_cloud


def generate_random_samples():
    args = parse_args_and_set_seed(task='gen_viz', description='Generate, visualize and render samples')
    assert args.model_head == 'VQVAE', 'Only VQVAE models can do random generation'
    model = get_model(**vars(args))
    datasets = dict(train_loader=EmptyDataset())
    trainer = get_trainer(model, datasets, args=args)
    trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    samples = model.random_sampling(args.gen)['recon'].cpu()
    for i, sample in enumerate(samples):
        if args.interactive_plot:
            show_pc(sample)
        sample_name = '_'.join(args.select_classes + [str(i)])
        render_cloud([sample], name=f'{sample_name}.png')


if __name__ == '__main__':
    generate_random_samples()
