"""Download a model published with the paper.

The archives are assets of a GitHub release. Each extracts in the folder of its
configuration, where the evaluation scripts find it.

Usage::

    uv run download_checkpoints.py --config-name best_airplane
    uv run download_checkpoints.py --config-name best_car
    uv run download_checkpoints.py --config-name best_chair
"""

from src.config import ExperimentConfig, hydra_main
from src.utils.download import download_extract_zip

RELEASE_URL = 'https://github.com/nverchev/PCGen/releases/download/published-models'


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Download and extract the published model of the configuration."""
    config_name = cfg.user.hydra.config_name
    download_extract_zip(cfg.par_dir, f'{RELEASE_URL}/{config_name}.zip')
    return


if __name__ == '__main__':
    main()
