from src.plot_PC import pc_show, render_cloud
from main import main


def visualize_reconstruction():
    for i, pc_in, pc_out in zip(*main(task='visualise reconstructions')):
        #pc_show(pc_in)
        #pc_show(pc_out)
        # render_cloud([pc_in.detach().cpu(), pc_out.detach().cpu()], name=f'reconstruction_{i}.png')
        render_cloud([pc_in.detach().cpu()], name=f'sample_{i}.png')
        render_cloud([ pc_out.detach().cpu()], name=f'reconstruction_{i}.png')


if __name__ == '__main__':
    visualize_reconstruction()
