from src.options import parse_args_and_set_seed
from src.dataset import EmptyDataset
from src.model import get_model
from src.trainer import get_trainer
from src.plot_PC import pc_show, render_cloud


def generate_random_samples():
    args = parse_args_and_set_seed(description='Generate, visualize and render samples from a loaded model')
    assert args.model_head == 'VQVAE', 'Only VQVAE models can do random generation'
    model = get_model(**vars(args))
    datasets = dict(train_loader=EmptyDataset())
    trainer = get_trainer(model, datasets, args=args)
    if args.load == 0:
        trainer.load()
    elif args.load > 0:
        trainer.load(args.load)
    samples = model.random_sampling(args.gen)['recon'].cpu()
    for i, sample in enumerate(samples):
        if args.interactive_plot:
            pc_show(sample)
        render_cloud([sample], name=f'generated_{i}.png')


if __name__ == '__main__':
    generate_random_samples()
