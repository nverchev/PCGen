from src.plot_PC import pc_show
from main import main


def generate_random_samples():
    loss_array = main(task='evaluate random generation')

if __name__ == '__main__':
    generate_random_samples()