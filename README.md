# PCGen

**PCGen** is a fully parallelizable generative model for point clouds. It decodes every point independently of the others, so it can generate a cloud of any size in a single pass, which makes it fast enough for real-time applications such as augmented and virtual reality.

This repository contains the code of the article [*PCGen: a fully parallelizable point cloud generative model*](https://doi.org/10.3390/s24051414), together with the published models, which you can evaluate without training.

The project uses the [DRYTorch](https://github.com/nverchev/drytorch) engine for ML experimentation.

---

## The PCGen Method

Generating point clouds is hard because a cloud is an unordered set, and most models scale poorly with the number of points. PCGen addresses this with a double autoencoder:

1. **Point autoencoder:** An encoder compresses a cloud into a codeword $w$, which is divided into chunks. Each chunk is replaced by its closest entry in a codebook, as in a VQ-VAE, giving the quantised codeword $\bar{w}$.
2. **Generator:** The decoder maps points drawn from a hypersphere to the surface of the object, conditioned on $\bar{w}$. Each point passes through the same shared weights, so the cost grows linearly with the number of points, and the points can be computed in parallel.
3. **Components:** Several components process the same point with different weights, so that each can model a part of the shape. An attention mechanism computed for every point mixes them.
4. **Filter:** A final graph filter rearranges the points more evenly on the surface.
5. **Codeword autoencoder:** A second variational autoencoder learns the distribution of the codewords in a continuous latent space with a VampPrior. Drawing from it and quantising the result gives new codewords, so the two models together generate new clouds.

The two autoencoders are trained one after the other: the second continues the run of the first.

---

## Project structure

```
PCGen/
├── configs/experiment/           # Hydra configurations, one entry for each setting
├── configs/tuning/               # Search spaces of the two phases
├── external/                     # CUDA matching extension for the earth mover distance
├── src/
│   ├── config/                   # Typed specifications and Hydra setup
│   ├── data/                     # Datasets, splits and derived datasets
│   ├── lib/                      # DRYTorch components: models, objectives, hooks, loaders, events
│   ├── module/                   # Encoders, decoders, autoencoders, layers
│   ├── trackers/                 # Experiment trackers, including the PyVista renderer
│   └── utils/                    # Distances, neighbour operations, generative scores
├── download_checkpoints.py       # Published models
├── train_autoencoder.py          # Phase one
├── train_w_autoencoder.py        # Phase two
├── evaluate_autoencoder.py       # Reconstructions
├── evaluate_w_autoencoder.py     # Codeword autoencoder
├── evaluate_double_reconstruction.py  # Reconstructions through the prior
├── evaluate_generation.py        # Generative scores
├── generate.py                   # Render new samples
├── visualize_reconstructions.py
├── visualize_double_reconstructions.py
├── profile_autoencoder.py        # Inference cost
├── tune_autoencoder.py           # Tuning of phase one
└── tune_w_autoencoder.py         # Tuning of phase two
```

---

## Installation

```bash
uv sync                      # minimal
uv sync --all-extras --dev   # trackers, tuning, visualization and development
uv sync --extra emd          # builds the CUDA matching extension, see external/README.md
```

The extension is optional: without it, the earth mover distance falls back to an exact but slow matching, which is unusable for a whole epoch.

### Configure the paths

Paths come from a gitignored `.env`:

```
DATA_DIR=/path/to/datasets
ROOT_EXP_DIR=./experiments
```

Datasets download and preprocess themselves on first use. To contribute, install the hooks with `pre-commit install`.

---

## Published Models

The models behind the results of the article are available as assets of the [`published-models`](https://github.com/nverchev/PCGen/releases/tag/published-models) release, one archive of about 15 MB for each class:

```bash
uv run download_checkpoints.py --config-name best_airplane
uv run download_checkpoints.py --config-name best_car
uv run download_checkpoints.py --config-name best_chair
```

Each archive holds the point autoencoder at epoch 1000 and the codeword autoencoder at epoch 500, converted to this code. They extract under `experiments/`, where the configurations `best_airplane`, `best_car` and `best_chair` find them:

```bash
uv run evaluate_autoencoder.py --config-name best_airplane            # reconstructions
uv run evaluate_double_reconstruction.py --config-name best_airplane  # reconstructions through the prior
uv run evaluate_generation.py --config-name best_airplane             # generative scores
uv run generate.py --config-name best_airplane                        # render new samples
```

---

## Usage

### Experiments

There are two settings, each a configuration of its own:

| Setting | Data | Head | Epochs |
|---|---|---|---|
| `ae` | ShapeNetAtlas, 13 classes, 2500 points | AE, Chamfer | 350 |
| `vqvae_<class>` | ShapeNetFlow, one class, 2048 points, resampled | VQVAE, Chamfer and EMD | 1000 |

The autoencoder alone reconstructs clouds. Only the quantised head can generate them, once its codeword autoencoder is trained.

### Training

```bash
# the plain autoencoder
uv run train_autoencoder.py --config-name ae
uv run evaluate_autoencoder.py --config-name ae

# the generative model: vqvae_airplane is the default, vqvae_car and vqvae_chair the other classes
uv run train_autoencoder.py                # phase one
uv run evaluate_autoencoder.py
uv run train_w_autoencoder.py              # phase two, continues the run of phase one
uv run evaluate_w_autoencoder.py
uv run evaluate_double_reconstruction.py   # reconstructions through the prior
uv run evaluate_generation.py              # generative scores
uv run generate.py                         # render samples
```

Run every phase with the same overrides, apart from `user.*`: they form the run name, and a later phase finds the earlier one under it.

### Oracle

The `Oracle` head gives the reference values. In `evaluate_autoencoder.py`, it measures the distance between two resamplings of the same cloud. In `evaluate_generation.py`, it draws real clouds from the training set, scoring what a perfect generator would reach.

```bash
uv run evaluate_autoencoder.py autoencoding.model.model_head=oracle
uv run evaluate_generation.py autoencoding.model.model_head=oracle
```

### Evaluation Protocol

The values in this README and in the article are computed without `final`, following the PointFlow and ShapeGF protocol: the models train on the training split, and both the reconstructions and the generative scores are measured on the validation split.

With `final=true`, the models train on the training and validation splits and are measured on the test split, so the values are not comparable: the test reconstructions are harder, and the test split is larger, which lowers the minimum matching distance of the same generator.

Reconstruction errors are measured on the original coordinates, which `de_normalize` restores by undoing the per-shape scaling of each cloud. Generative scores compare clouds scaled to the unit ball, each generated one being normalized the way the dataset is.

### Visualisation and Profiling

- `visualize_reconstructions.py` renders reconstructions of the sample indices in `user.plot.sample_indices`, and `visualize_double_reconstructions.py` renders those through the prior.
- `profile_autoencoder.py` reports the cost of generating a batch of clouds, or of reconstructing one with the plain autoencoder.

### Tuning

`tune_autoencoder.py` and `tune_w_autoencoder.py` search the settings of the two phases with Optuna. Each trial trains for `n_epochs` (100) from the configuration named by `config_name` (`vqvae_airplane`) and scores the validation loss, so trials never use the test split. The tuned settings form a scheme in `configs/tuning/<phase>/tune`, chosen with `tune=`.

```bash
uv run tune_autoencoder.py tune=embedding        # learn, embedding, architecture
uv run tune_w_autoencoder.py tune=model          # learn, model, loss
```

The codeword trials encode the autoencoder already trained with the configuration, so train it first with the same `config_name` and without `final=true`. Studies are stored in `optimization_db/` and continue when run again.

---

## Results

Measured on the validation split with the published models, as described in [Evaluation Protocol](#evaluation-protocol).

### Reconstruction

**CD** is the Chamfer distance and **EMD** the earth mover distance. The double reconstruction passes through the codeword autoencoder too. The oracle compares two resamplings of the same cloud, which is the error that sampling alone causes.

| Class | CD ×10⁴ | EMD ×10² | Double CD ×10⁴ | Double EMD ×10² | Oracle CD ×10⁴ | Oracle EMD ×10² |
|---|---|---|---|---|---|---|
| Airplane | 0.87 | 2.20 | 1.04 | 2.38 | 0.84 | 2.07 |
| Car | 4.84 | 3.63 | 6.22 | 4.19 | 3.90 | 3.28 |
| Chair | 5.56 | 4.00 | 11.23 | 5.13 | 3.20 | 3.30 |

### Generation

Scores of the generated clouds against the validation split with the Chamfer distance: minimum matching distance (**MMD**, lower is better), coverage (**COV**, higher is better) and 1-nearest neighbour accuracy (**1-NNA**, lower is better). This code gives the mean over the repetitions of the score, and the article gives the mean of its runs.

| Class | MMD ×10³ | COV | 1-NNA | Article: MMD ×10³ | Article: COV | Article: 1-NNA |
|---|---|---|---|---|---|---|
| Airplane | 0.93 | 43.9 % | 72.3 % | 0.90 | 44.3 % | 70.9 % |
| Car | 3.34 | 46.3 % | 59.6 % | 3.34 | 45.3 % | 59.4 % |
| Chair | 6.95 | 46.2 % | 60.3 % | 6.92 | 45.7 % | 60.3 % |

### Key Findings

- **Real-time generation:** A cloud is drawn in about 25 ms in the article's measurements: 24.25 ms for the generator and 0.89 ms for the codeword decoder.
- **Competitive quality:** In the article, PCGen matches or improves on slower methods in reconstruction and generation, as detailed in its tables.
- **Reproducible results:** The published models, converted to this code, give the same reconstruction and generative scores as the article, up to the sampling of the clouds.

---

## Citation

```bibtex
@article{vercheval2024pcgen,
  title={{PCGen}: a fully parallelizable point cloud generative model},
  author={Vercheval, Nicolas and Royen, Remco and Pi{\v{z}}urica, Aleksandra and Munteanu, Adrian},
  journal={Sensors},
  volume={24},
  number={5},
  pages={1414},
  year={2024},
  doi={10.3390/s24051414}
}
```
