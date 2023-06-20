import plotly.graph_objects as go
import numpy as np
import torch
from simple_3dviz import Spherecloud
from simple_3dviz import Scene
from simple_3dviz.utils import save_frame

#import pyvista as pv

import os


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 5, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(scene=dict(aspectmode='data'),
                                     updatemenus=[dict(type='buttons',
                                                       showactive=False,
                                                       y=1,
                                                       x=0.8,
                                                       xanchor='left',
                                                       yanchor='bottom',
                                                       pad=dict(t=45, r=10),
                                                       buttons=[dict(label='Play',
                                                                     method='animate',
                                                                     args=[None,
                                                                           dict(frame=dict(duration=50, redraw=True),
                                                                                transition=dict(duration=0),
                                                                                fromcurrent=True,
                                                                                mode='immediate'
                                                                                )]
                                                                     )
                                                                ]
                                                       )
                                                  ]
                                     ),
                    frames=frames
                    )

    return fig


def show_pc(pcs, colors=None, colorscale='blue' + 'red'):
    if not isinstance(pcs, tuple) and not isinstance(pcs, list):
        pcs = (pcs, )
        colors = (colors, )
    data = []
    for pc, color in zip(pcs, colors):
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().detach().numpy()
        xs, zs, ys = pc.transpose()
        data.append(go.Scatter3d(x=xs, y=-ys, z=zs,
                                 mode='markers',
                                 marker=dict(size=2, color=color, colorscale=colorscale)))
    fig = visualize_rotate(data)
    fig.show()


def render_cloud(clouds, name, colors=iter(lambda: 0, 1), colorscale='sequence'):
    scene = Scene(background=(.8, .9, 0.9, .9), size=(1024, 1024))
    blue = np.array([0.3, 0.3, 0.9, 0.9])
    red = np.array([0.9, 0.3, 0.3, 0.9])
    green = np.array([0.3, 0.9, 0.3, 0.9])
    violet = np.array([0.6, 0.0, 0.9, 0.9])
    orange = np.array([0.9, 0.6, 0.0, 0.9])
    color_sequence = [blue, red,  green, violet, orange]
    l = len(clouds)
    for i, (cloud, color) in enumerate(zip(clouds, colors)):
        if not len(cloud):
            continue
        if colorscale == 'blue' + 'red':
            color = np.ones(cloud.shape[0])[:, None] * ((1 - color) * blue + color * red)[None, :]
        if colorscale == 'sequence':
            color = np.ones(cloud.shape[0])[:, None] * color_sequence[color][None, :]

        sizes = np.ones(cloud.shape[0]) * 0.02
        #cloud = cloud + np.array([(1 - l) / 2 + i, 0, 0])[None, :]
        s = Spherecloud(cloud, sizes=sizes, colors=color)
        scene.add(s)
    scene.camera_position = (-1.2, 1, -2)
    scene.up_vector = (0, 1, 0)
    scene.light = (60, 60, 60)
    scene.render()
    if not os.path.exists('images'):
        os.mkdir('images')
    save_frame(os.path.join('images', name), scene.frame)



# def render_cloud(clouds, name, colors=iter(lambda: 0, 1), colorscale='sequence'):
#     plotter = pv.Plotter(lighting='none', window_size=(1024, 1024))
#     plotter.set_background(color=(.8, .9, 0.9, .9))
#     blue = np.array([0.3, 0.3, 0.9, 0.9])
#     red = np.array([0.9, 0.3, 0.3, 0.9])
#     green = np.array([0.3, 0.9, 0.3, 0.9])
#     violet = np.array([0.6, 0.0, 0.9, 0.9])
#     orange = np.array([0.9, 0.6, 0.0, 0.9])
#     color_sequence = [blue, red,  green, violet, orange]
#     l = len(clouds)
#     for i, (cloud, color) in enumerate(zip(clouds, colors)):
#         if not len(cloud):
#             continue
#         if colorscale == 'blue' + 'red':
#             color = np.ones(cloud.shape[0])[:, None] * ((1 - color) * blue + color * red)[None, :]
#         if colorscale == 'sequence':
#             color = np.ones(cloud.shape[0])[:, None] * color_sequence[color][None, :]
#
#         sizes = np.ones(cloud.shape[0]) * 0.02
#         #cloud = cloud + np.array([(1 - l) / 2 + i, 0, 0])[None, :]
#         plotter.add_mesh(pv.PolyData(cloud), point_size=sizes, color=color)
#         scene.add(s)
#     scene.camera_position = (-1.2, 1, -2)
#     scene.up_vector = (0, 1, 0)
#     scene.light = (60, 60, 60)
#     scene.render()
#     if not os.path.exists('images'):
#         os.mkdir('images')
#     save_frame(os.path.join('images', name), scene.frame)


