import plotly.graph_objects as go
import numpy as np


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
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                          showactive=False,
                                          y=1,
                                          x=0.8,
                                          xanchor='left',
                                          yanchor='bottom',
                                          pad=dict(t=45, r=10),
                                          buttons=[dict(label='Play',
                                                        method='animate',
                                                        args=[None, dict(frame=dict(duration=50, redraw=True),
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


def pcshow(pcs, colors=None):
    if not isinstance(pcs, list):
      pcs = [pcs]
      colors = [colors]
    data = []
    for pc, color in zip(pcs, colors):
      pc = pc.cpu().detach().numpy()
      xs, zs, ys = pc.transpose()
      data.append(go.Scatter3d(x=xs, y=-ys, z=zs,
                          mode='markers',
                          marker=dict(color = color, colorscale = 'bluered')))
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                                  line=dict(width=0.2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
