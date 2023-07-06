import torch

from src.options import parse_args_and_set_seed
from src.dataset import EmptyDataset
from src.model import get_model
from src.trainer import get_trainer
from src.viz_pc import render_cloud
from src.neighbour_op import graph_filtering


def generate_random_samples():
    args = parse_args_and_set_seed(task='gen_viz', description='Generate, visualize and render samples')
    assert args.model_head == 'VQVAE', 'Only VQVAE models can do random generation'
    model = get_model(**vars(args))
    datasets = dict(train_loader=EmptyDataset())
    trainer = get_trainer(model, datasets, args=args)
    trainer.load(args.load_checkpoint if args.load_checkpoint else None)
    s = torch.randn(args.gen, args.sample_dim, args.m_test, device=args.device)
    z_bias = torch.zeros(args.gen, args.z_dim, device=args.device)
    z_bias[:, args.bias_dim] = args.bias_value
    att = None
    components = None
    if args.add_viz == 'sampling_loop':
        bbox1 = torch.eye(args.sample_dim, device=args.device)
        bbox2 = -torch.eye(args.sample_dim, device=args.device)
        bbox = torch.cat((bbox1, bbox2.flip(1)), dim=1).unsqueeze(0).expand(args.gen, -1, -1)
        s = torch.cat([s] + [t * bbox + (1 - t) * bbox.roll(1, dims=2) for t in torch.linspace(0, 1, 30)], dim=2)
    elif args.add_viz == 'components':
        att = torch.empty(args.gen, args.m_test, args.components, device=args.device)
        components = torch.empty(args.gen, 3, args.m_test, args.components)
    elif args.add_viz == 'filter':
        trainer.model.decoder.filtering = False
    elif args.add_viz == 'none':
        pass
    elif args.add_viz:
        raise ValueError(f'{args.add_vix} is not a recognized argument')
    samples_and_loop = model.random_sampling(args.gen, s, att, components, z_bias)['recon'].cpu()
    samples, *loops = samples_and_loop.split(args.m_test, dim=1)

    for i, sample in enumerate(samples):
        if args.add_viz == 'sampling_loop':
            sample_name = '_'.join(args.select_classes + ['sampling_loop', str(i)])
            render_cloud((sample, loops[0][i]), name=f'{sample_name}.png', interactive=args.interactive_plot)
        elif args.add_viz == 'components':
            threshold = 0.  # boundary points shown in blue
            att_max, att_argmax = att[i].max(dim=1)
            indices = (att_argmax.cpu() + 1) * (att_max > threshold).bool().cpu()
            pc_list = [sample[indices == component] for component in range(args.components + 1)]
            sample_name = '_'.join(args.select_classes + ['attention', str(i)])
            render_cloud(pc_list,  name=f'{sample_name}.png', interactive=args.interactive_plot)
            component = components[i].cpu().transpose(1, 0)
            components_cloud = []
            for j, j_component in enumerate(component.unbind(2)):
                components_cloud.append(j_component + torch.FloatTensor([[(1 - args.components) / 2 + j, 0, 0]]))
            components_cloud = torch.cat(components_cloud, dim=0)
            components_cloud /= args.components
            sample_name = '_'.join(args.select_classes + ['components', str(i)])
            render_cloud((components_cloud, ),  name=f'{sample_name}.png', interactive=args.interactive_plot)
        elif args.add_viz == 'filter':
            filter_direction = graph_filtering(sample.transpose(0,1).unsqueeze(0)).squeeze().transpose(0, 1) - sample
            sample_name = '_'.join(args.select_classes + ['filter', str(i)])
            render_cloud((sample, ), name=f'{sample_name}.png', arrows=filter_direction,
                         interactive=args.interactive_plot)

        elif args.add_viz == 'none':
            sample_name = '_'.join(args.select_classes + [str(i)])
            render_cloud((sample, ), name=f'{sample_name}.png', interactive=args.interactive_plot)
            pass


if __name__ == '__main__':
    generate_random_samples()
