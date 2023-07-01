import os
import numpy as np
import pyvista as pv


def render_cloud(clouds, name, colorscale='sequence', interactive=True, arrows=None):
    blue = np.array([0.3, 0.3, 0.9])
    red = np.array([0.9, 0.3, 0.3])
    green = np.array([0.3, 0.9, 0.3])
    violet = np.array([0.6, 0.0, 0.9])
    orange = np.array([0.9, 0.6, 0.0])
    color_sequence = [blue, red,  green, violet, orange]
    plotter = pv.Plotter(lighting='three_lights', window_size=(1024, 1024), notebook=False, off_screen=not interactive)
    plotter.camera_position = pv.CameraPosition((-3, -2, 1), focal_point=(0, 0, 0), viewup=(0, 0, 1))
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
        cloud = pv.PolyData(cloud[:, [0, 2, 1]].numpy())
        plotter.add_mesh(cloud, color=color, point_size=15, render_points_as_spheres=True, smooth_shading=True)
        if arrows is not None:
            geom = pv.Arrow(shaft_radius=0.1, tip_radius=0.2, scale='auto')
            cloud["vectors"] = arrows
            cloud.set_active_vectors("vectors")
            arrows_glyph = cloud.glyph(orient="vectors",  geom=geom)
            plotter.add_mesh(arrows_glyph, lighting=True, line_width=10, color=red, show_scalar_bar=False, edge_color=red)

    plotter.enable_eye_dome_lighting()
    plotter.enable_shadows()
    if interactive:
        plotter.set_background(color='white')
        plotter.show()
    else:
        if not os.path.exists('images'):
            os.mkdir('images')
        plotter.screenshot(os.path.join('images', name), window_size=(1024, 1024), transparent_background=True)
    plotter.close()



