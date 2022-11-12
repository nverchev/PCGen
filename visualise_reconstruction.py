from src.plot_PC import pc_show
from main import main


def visualize_reconstruction():
    for pc_in, pc_out in zip(*main(task='visualise reconstructions')):
        pc_show(pc_in)
        pc_show(pc_out)


if __name__ == '__main__':
    visualize_reconstruction()
