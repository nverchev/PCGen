from src.plot_PC import pc_show, render_cloud
from main import main
import numpy as np
import torch


def generate_random_samples():
    model = main(task='return loaded model for random generation')
    with torch.no_grad():
        samples = model.random_sampling(16)['recon'].cpu()
    np.save('generated_clouds', samples)
    for i, sample in enumerate(samples):
        #pc_show(sample)
        render_cloud([sample], name=f'generated_{i}.png')

if __name__ == '__main__':
    generate_random_samples()
