import torch

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
    s = torch.randn(args.gen, args.sample_dim, args.m_test, device=args.device)
    att = None
    if args.add_viz == 'sampling_loop':
        bbox1 = torch.eye(args.sample_dim, device=args.device)
        bbox2 = -torch.eye(args.sample_dim, device=args.device)
        bbox = torch.cat((bbox1, bbox2.flip(1)), dim=1).unsqueeze(0).expand(args.gen, -1, -1)
        s = torch.cat([s] + [t * bbox + (1 - t) * bbox.roll(1, dims=2) for t in torch.linspace(0, 1, 30)], dim=2)
    elif args.add_viz == 'components':
        att = torch.empty(args.gen, args.m_test, args.components, device=args.device)
    elif args.add_viz == 'filter':
        trainer.model.decoder.filtering = False
    elif args.add_viz:
        raise ValueError(f'{args.add_vix} is not a recognized argument')
    samples_and_diameters = model.random_sampling(args.gen, s, att)['recon'].cpu()
    samples, *diameters = samples_and_diameters.split(args.m_test, dim=1)

    for i, sample in enumerate(samples):
        if args.add_viz == 'sampling_loop':
            if args.interactive_plot:
                show_pc((sample, diameters[0][i]), colors=[0, 1])
            sample_name = '_'.join(args.select_classes + [str(i)])
            render_cloud((sample, diameters[0][i]), colors=[0, 1], name=f'{sample_name}.png')
        elif args.add_viz == 'components':
            threshold = 0.6  # boundary points shown in blue
            att_max, att_argmax = att.max(dim=2)
            indices = (att_argmax.cpu() + 1) * (att_max > threshold).bool().cpu()
            pc_list = [samples[indices == component] for component in range(args.components + 1)]
            if args.interactive_plot:
                show_pc(pc_list, colors=range(args.components + 1))
            sample_name = '_'.join(args.select_classes + [str(i)])
            render_cloud(pc_list, colors=range(args.components + 1), name=f'{sample_name}.png')
        elif args.add_viz == 'filter':
            pass



if __name__ == '__main__':
    generate_random_samples()
