"""Report of the cost of reconstructing or generating clouds."""

import statistics
import time

from collections.abc import Sequence

import torch

from deepspeed.profiling.flops_profiler import get_model_profile
from torch import nn

from src.config import Experiment, ExperimentConfig, hydra_main
from src.data.structures import Inputs
from src.module import AE, VQVAE, VQVAEWithVAE, WAutoEncoder
from src.lib import get_autoencoder_model
from src.module.generation import Drawing

N_TIMED_PASSES = 50
N_WARM_UP_PASSES = 10


@torch.inference_mode()
def profile_autoencoder() -> None:
    """Report the operations and the memory a batch of clouds costs.

    The quantized head is profiled on generation: drawing the latent variable
    from the prior, decoding it into codes and decoding the cloud. The
    standard autoencoder is profiled on reconstruction: encoding a cloud of
    ``input_points`` points and decoding it. The initial points on the
    generating surface are drawn beforehand, so neither profile includes
    them. The weights are not trained, as the cost does not depend on them.

    The latency is timed apart from the profile, without its hooks, over
    several passes after the warm-up.

    Raises:
        ValueError: if the configured head does not support profiling.

    Side Effects:
        Prints the profile and the latency to the standard output.
    """
    cfg = Experiment.get_config()
    autoencoder = get_autoencoder_model()
    module = autoencoder.module.eval()
    device = autoencoder.device
    batch_size = cfg.autoencoding.train.batch_size
    args: list[object]
    model: nn.Module
    if isinstance(module, VQVAE):
        model = Drawing(VQVAEWithVAE(module, WAutoEncoder().to(device)))
        args = [batch_size, module.get_sampling(batch_size, loop=False)]
    elif isinstance(module, AE):
        cloud = torch.ones(batch_size, cfg.data.dataset.input_points, 3, device=device)
        model = module
        args = [Inputs(cloud=cloud, initial_sampling=module.get_sampling(batch_size, loop=False))]
    else:
        msg = f'Model {type(module).__name__} does not support profiling.'
        raise ValueError(msg)

    get_model_profile(
        model=model,
        args=args,
        print_profile=True,
        detailed=True,
        module_depth=2,
        top_modules=2,
        warm_up=N_WARM_UP_PASSES,
        as_string=True,
    )
    latencies = _time_passes(model, args, device)
    quartiles = statistics.quantiles(latencies, n=4)
    print(
        f'Latency over {N_TIMED_PASSES} passes: median {statistics.median(latencies):.2f} ms '
        f'(quartiles {quartiles[0]:.2f}-{quartiles[2]:.2f} ms)'
    )
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for profiling."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    with exp.create_run(record=False):
        profile_autoencoder()

    return


def _time_passes(model: nn.Module, args: Sequence[object], device: torch.device) -> list[float]:
    for _ in range(N_WARM_UP_PASSES):
        model(*args)

    latencies: list[float] = []
    for _ in range(N_TIMED_PASSES):
        if device.type == 'cuda':
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(*args)
            end.record()
            torch.cuda.synchronize(device)
            latencies.append(start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            model(*args)
            latencies.append(1000 * (time.perf_counter() - start_time))

    return latencies


if __name__ == '__main__':
    main()
