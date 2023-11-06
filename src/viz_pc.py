import os
import numpy as np
import torch
# Using pyvista instead of plotly because of its ray casting. No current integration with visdom.
import pyvista as pv
from src.neighbour_op import graph_filtering


def render_cloud(clouds, name, colorscale='sequence', interactive=True, arrows=None):
    blue = np.array([0.3, 0.3, 0.9])
    red = np.array([0.9, 0.3, 0.3])
    green = np.array([0.3, 0.9, 0.3])
    violet = np.array([0.6, 0.0, 0.9])
    orange = np.array([0.9, 0.6, 0.0])
    color_sequence = [blue, red, green, violet, orange]
    plotter = pv.Plotter(lighting='three_lights', window_size=(1024, 1024), notebook=False, off_screen=not interactive)
    plotter.camera_position = pv.CameraPosition((-3., -2.1, 1.1), focal_point=(0, 0, 0), viewup=(0, 0, 1))
    plotter.camera_position = pv.CameraPosition((2, 4, 0), focal_point=(0, 0, 0), viewup=(0, 0, 1))

    for i in [-1, 1]:
        light_point = (i, 1, 0)
        light = pv.Light(position=light_point, focal_point=(0, 0, 0), intensity=.2, positional=True)
        plotter.add_light(light)
    for i, cloud in enumerate(clouds):
        if not len(cloud):
            continue
        if colorscale == 'blue' + 'red':
            color = (1 - i) * blue + i * red
        elif colorscale == 'sequence':
            color = color_sequence[i]
        else:
            raise ValueError('Colorscale not available')
        n = cloud.shape[0]
        cloud = pv.PolyData(cloud[:, [0, 2, 1]].numpy())
        geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
        cloud["radius"] = .01 * np.ones(n)
        glyphed = cloud.glyph(scale='radius', geom=geom, orient=False)
        plotter.add_mesh(glyphed, color=color, point_size=15, render_points_as_spheres=True, smooth_shading=True)
        if arrows is not None:
            geom = pv.Arrow(shaft_radius=.1, tip_radius=.2, scale=1)
            cloud["vectors"] = arrows[:, [0, 2, 1]]
            cloud.set_active_vectors("vectors")
            arrows_glyph = cloud.glyph(orient="vectors", geom=geom)
            plotter.add_mesh(arrows_glyph, lighting=True, line_width=10, color=red, show_scalar_bar=False,
                             edge_color=red)
    #plotter.enable_eye_dome_lighting()
    #plotter.enable_shadows()
    if interactive:
        plotter.set_background(color='white')
        plotter.show()
    else:
        if not os.path.exists('images'):
            os.mkdir('images')
        plotter.screenshot(os.path.join('images', name), window_size=(1024, 1024), transparent_background=True)
    plotter.close()


def infer_and_visualize(model, args, n_clouds, mode='recon', z_bias=None, input_pc=None):
    s = torch.randn(n_clouds, args.sample_dim, args.m_test, device=args.device)
    att = None
    components = None
    if args.add_viz == 'sampling_loop':
        bbox1 = torch.eye(args.sample_dim, device=args.device, dtype=torch.float32)
        bbox2 = -torch.eye(args.sample_dim, device=args.device, dtype=torch.float32)
        bbox = torch.cat((bbox1, bbox2), dim=1).unsqueeze(0).expand(n_clouds, -1, -1)
        s = torch.cat([s] + [t * bbox.roll(1, dims=2) + (1 - t) * bbox for t in torch.arange(0, 1, 0.03)], dim=2)
        model.decoder.filtering = False
    elif args.add_viz == 'components':
        att = torch.empty(n_clouds, args.m_test, args.components, device=args.device)
        components = torch.empty(n_clouds, 3, args.m_test, args.components)
    elif args.add_viz == 'filter':
        model.decoder.filtering = False
    elif args.add_viz == 'none':
        pass
    elif args.add_viz:
        raise ValueError(f'{args.add_vix} is not a recognized argument')
    if mode == 'recon':
        assert z_bias is None
        assert input_pc is not None
        with torch.inference_mode():
            model.eval()
            samples_and_loop = model(input_pc, None, s, att, components)
    elif mode == 'gen':
        assert z_bias is not None
        assert input_pc is None
        samples_and_loop = model.random_sampling(n_clouds, s, att, components, z_bias)
    else:
        raise ValueError('Mode can only be "recon" or "gen"')
    samples_and_loop = samples_and_loop['recon'].cpu()
    samples, *loops = samples_and_loop.split(args.m_test, dim=1)

    def naming_syntax(num, viz_name=None):
        if mode == 'recon':
            num = args.viz[num]
        viz_name = [viz_name] if viz_name else []
        return '_'.join([mode] + args.select_classes + viz_name + [str(num)])

    for i, sample in enumerate(samples):

        if args.add_viz == 'sampling_loop':
            sample_name = naming_syntax(i, 'sampling_loop')
            render_cloud((sample, loops[0][i]), name=f'{sample_name}.png', interactive=args.interactive_plot)
        elif args.add_viz == 'components':
            threshold = 0.  # boundary points shown in blue
            att_max, att_argmax = att[i].max(dim=1)
            indices = (att_argmax.cpu() + 1) * (att_max > threshold).bool().cpu()
            pc_list = [sample[indices == component] for component in range(args.components + 1)]
            sample_name = naming_syntax(i, 'attention')
            render_cloud(pc_list, name=f'{sample_name}.png', interactive=args.interactive_plot)
            component = components[i].cpu().transpose(1, 0)
            components_cloud = [[]]
            for j, j_component in enumerate(component.unbind(2)):
                j_component = j_component + torch.FloatTensor([[(1 - args.components) / 2 + j, 0, 0]])
                components_cloud.append(j_component/args.components)
            sample_name = naming_syntax(i, 'components')
            render_cloud(components_cloud, name=f'{sample_name}.png', interactive=args.interactive_plot)
        elif args.add_viz == 'filter':
            filter_direction = graph_filtering(sample.transpose(0, 1).unsqueeze(0)).squeeze().transpose(0, 1) - sample
            sample_name = naming_syntax(i, 'filter')
            render_cloud((sample, ), name=f'{sample_name}.png', arrows=filter_direction,
                         interactive=args.interactive_plot)
        elif args.add_viz == 'none':
            sample_name = naming_syntax(i)
            render_cloud((sample,), name=f'{sample_name}.png', interactive=args.interactive_plot)
            pass
