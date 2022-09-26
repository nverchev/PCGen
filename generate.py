from src.plot_PC import pc_show
from main import main

def generate_random_samples():
    model, z = main(task='return loaded model for random generation')
    samples = model.decode_z({'z': z})['recon']
    for sample in samples:
        pc_show(sample)

if __name__ == '__main__':
    generate_random_samples()
