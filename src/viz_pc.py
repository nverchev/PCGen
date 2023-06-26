import os
import numpy as np
import pyvista as pv


def render_cloud(clouds, name, colors=iter(lambda: 0, 1), colorscale='sequence', interactive=True):
    blue = np.array([0.3, 0.3, 0.9])
    red = np.array([0.9, 0.3, 0.3])
    green = np.array([0.3, 0.9, 0.3])
    violet = np.array([0.6, 0.0, 0.9])
    orange = np.array([0.9, 0.6, 0.0])
    color_sequence = [blue, red,  green, violet, orange]
    light_point = (1, 1, 5)

    plotter = pv.Plotter(lighting='three_lights', window_size=(1024, 1024), notebook=False, off_screen=not interactive)
    plotter.camera_position = pv.CameraPosition((-3, -2, 1), focal_point=(0, 0, 0), viewup=(0, 0, 1))
    for i, (cloud, color) in enumerate(zip(clouds, colors)):
        if not len(cloud):
            continue
        if colorscale == 'blue' + 'red':
            color = np.ones(cloud.shape[0])[:, None] * ((1 - color) * blue + color * red)[None, :]
        if colorscale == 'sequence':
            color = np.ones(cloud.shape[0])[:, None] * color_sequence[color][None, :]

        sizes = np.ones(cloud.shape[0]) * 0.02
        #cloud = cloud + np.array([(1 - l) / 2 + i, 0, 0])[None, :]
        cloud = pv.PolyData(cloud[:, [0, 2, 1]].numpy())

        plotter.add_mesh(cloud, color=color[0], point_size=15, render_points_as_spheres=True, smooth_shading=True)
    plotter.set_background(color='white')
    plotter.enable_eye_dome_lighting()
    plotter.enable_shadows()
    if interactive:
        plotter.show()
    else:
        if not os.path.exists('images'):
            os.mkdir('images')
        plotter.screenshot(os.path.join('images', name), window_size=(1024, 1024), transparent_background=True)
    plotter.close()



