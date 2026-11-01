"""Specification for the configuration files.

Root config hierarchy::

    ExperimentConfig
    ├── autoencoding: AutoencodingConfig (PhaseConfig)
    │   ├── model: AutoencoderModelConfig
    │   ├── architecture: AutoencoderArchitectureConfig
    │   └── train: AutoencoderTrainingConfig
    │       ├── learn: LearningConfig
    │       │   └── scheduler: SchedulerConfig
    │       └── loss: AutoencoderLossConfig
    ├── w_autoencoding: WAutoencodingConfig (PhaseConfig)
    │   ├── architecture: WEncoderArchitectureConfig
    │   ├── model: WEncoderModelConfig
    │   └── train: WEncoderTrainingConfig
    │       ├── learn: LearningConfig
    │       │   └── scheduler: SchedulerConfig
    │       └── loss: WEncoderLossConfig
    ├── data: DataConfig
    │   └── dataset: DatasetConfig
    ├── evaluation: EvaluationConfig
    └── user: UserSettings
        ├── trackers: TrackerList
        ├── path: PathSpecs          ← populated from .env
        └── hydra: HydraSettings     ← populated at runtime
"""

import dataclasses
import pathlib
from functools import cached_property
from typing import Annotated, Any, Literal

import torch
from pydantic import BeforeValidator, Field, PlainValidator
from pydantic.dataclasses import dataclass

from src.config.environment import PROJECT_NAME, ROOT_DIR, VERSION, EnvSettings, get_current_branch
from src.config.options import (
    Datasets,
    Decoders,
    Encoders,
    ModelHeads,
    Overlays,
    ReconLosses,
    Schedulers,
    ShapenetFlowClasses,
)
from src.config.torch import (
    ActClass,
    get_activation_cls,
    get_device,
    get_mixed_precision,
    get_optim_cls,
    set_seed,
)


ActivationClass = Annotated[ActClass, BeforeValidator(get_activation_cls)]
Device = Annotated[torch.device | Literal['auto'], PlainValidator(get_device)]
OptimizerClass = Annotated[type[torch.optim.Optimizer], BeforeValidator(get_optim_cls)]
MixedPrecision = Annotated[torch.dtype | None, PlainValidator(get_mixed_precision)]
PositiveInt = Annotated[int, Field(ge=0)]
StrictlyPositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(ge=0.0)]
UnitIntervalFloat = Annotated[float, Field(ge=0.0, le=1.0)]
InputPointsInt = Annotated[int, Field(ge=4)]


# ---------------------------------------------------------------------------
# Optimisation & Learning configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    """Specification for the learning-rate scheduler.

    Attributes:
        function: Decay curve for the learning rate decay.
        min_decay: Fraction of the initial lr at the end of train.
        decay_steps: Epochs before lr decays stops.
    """

    function: Schedulers
    min_decay: UnitIntervalFloat
    decay_steps: PositiveInt


@dataclass(frozen=True)
class LearningConfig:
    """Specification for the optimiser and learning schedule.

    Attributes:
        optimizer: PyTorch optimizer, written by its name (see https://pytorch.org/docs/stable/optim.html).
        lr: Learning rate.
        weight_decay: Weight decay.
        scheduler: Learning-rate scheduler configuration.
    """

    optimizer: OptimizerClass
    lr: PositiveFloat
    weight_decay: PositiveFloat
    scheduler: SchedulerConfig


# ---------------------------------------------------------------------------
# Phase 1: Autoencoder configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoencoderModelConfig:
    """Model head and vector-quantization options.

    Attributes:
        model_head: Head type (AE, VQVAE, Oracle).
        book_size: Dictionary size.
        embedding_dim: Codes length.
        w_dim: Codeword length.
        vq_noise: Noise when redistributing the codes.
    """

    model_head: ModelHeads
    book_size: PositiveInt
    embedding_dim: PositiveInt
    w_dim: StrictlyPositiveInt
    vq_noise: PositiveFloat

    def __post_init__(self) -> None:
        """Check that the codeword divides into whole codes.

        Raises:
            ValueError: if the codeword is not as wide as the codes it holds.
        """
        if self.model_head != ModelHeads.VQVAE or self.embedding_dim <= 0:
            return

        remainder = self.w_dim % self.embedding_dim
        if remainder:
            msg = (
                f'A codeword of {self.w_dim} does not divide into codes of '
                f'{self.embedding_dim}: it holds {self.dim_codes} of them and {remainder} over.'
            )
            raise ValueError(msg)

        return

    @property
    def dim_codes(self) -> int:
        """Codes a codeword holds, which is its width over the width of one."""
        if self.embedding_dim == 0:
            return 0

        return self.w_dim // self.embedding_dim


@dataclass(frozen=True)
class EncoderConfig:
    """Architectural options for the point encoder.

    Attributes:
        name: Encoder name.
        n_neighbours: Number of neighbours of a point (counting the point itself) in (L)DGCNN.
    """

    name: Encoders
    n_neighbours: StrictlyPositiveInt


@dataclass(frozen=True)
class DecoderConfig:
    """Architectural options for the point decoder.

    Attributes:
        name: Decoder model.
        n_components: Number of components of PCGen or patches in AtlasNet.
        filtering: Allow filtering after decoder.
        sample_dim: Dimensions of the sampling sphere.
        hidden_dims: First dimension is number of channels for mapping the initial sampling,
            the second is a dummy variable later overwritten by w_dim, then channels for each component.
        attention_temperature: Temperature of the softmax choosing the component of every point.
    """

    name: Decoders
    n_components: StrictlyPositiveInt
    filtering: bool
    sample_dim: StrictlyPositiveInt
    hidden_dims: tuple[int, ...]
    attention_temperature: PositiveFloat


@dataclass(frozen=True)
class AutoencoderArchitectureConfig:
    """Architectural options for the autoencoder.

    Attributes:
        encoder: Architecture of the point encoder.
        decoder: Architecture of the point decoder.
        act_cls: Activation of the encoders and of the codeword encoder and decoder, written by its name;
            the PCGen generator uses ReLU. Defaults to ``LeakyReLU`` with slope 0.2.
        bn_momentum: Momentum for batch normalization layers.
    """

    encoder: EncoderConfig
    decoder: DecoderConfig
    act_cls: ActivationClass
    bn_momentum: UnitIntervalFloat


@dataclass(frozen=True)
class AutoencoderLossConfig:
    """Specification for autoencoder loss configuration and coefficients.

    Attributes:
        recon_loss: ChamferEMD adds both Chamfer and EMD.
        c_chamfer: Coefficient for Chamfer loss.
        c_emd: Coefficient for Earth Mover's Distance loss.
        c_commitment: Coefficient for commitment loss, relative to the reconstruction averaged over the points.
        c_embedding: Coefficient for embedding loss, relative to the reconstruction averaged over the points.
    """

    recon_loss: ReconLosses
    c_chamfer: PositiveFloat
    c_emd: PositiveFloat
    c_commitment: PositiveFloat
    c_embedding: PositiveFloat


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    """Specification for autoencoder training.

    Attributes:
        batch_size: Batch size.
        n_epochs: Number of total training epochs.
        learn: Optimiser and scheduler configuration.
        loss: Loss function and coefficient configuration.
        m_training: Points generated in training, 0 for input number of points.
        m_test: Points generated in testing, 0 for input number of points.
        de_normalize: Compare reconstruction and input on coordinates before normalization.
        checkpoint_every: Save a checkpoint every n epochs.
        mixed_precision: Autocast precision, ``float16`` or ``bfloat16``; empty for none.
        pruning_window: Number of last validations the pruner averages when tuning.
    """

    batch_size: StrictlyPositiveInt
    n_epochs: StrictlyPositiveInt
    learn: LearningConfig
    loss: AutoencoderLossConfig
    m_training: PositiveInt
    m_test: PositiveInt
    de_normalize: bool
    checkpoint_every: StrictlyPositiveInt
    mixed_precision: MixedPrecision
    pruning_window: StrictlyPositiveInt


@dataclass(frozen=True)
class PhaseConfig:
    """Specification for a single phase.

    Attributes:
        name: Name of the phase.
        architecture: Architectural choices for the phase.
        model: Model configuration.
        train: Training loop and learning schedule configuration.
    """

    name: str
    architecture: Any
    model: Any
    train: Any


@dataclass(frozen=True)
class AutoencodingConfig(PhaseConfig):
    """Specification for the autoencoder phase.

    Attributes:
        name: Name of the phase.
        model: Model head and vector-quantization options.
        architecture: Architectural choices for encoder and decoder.
        train: Training hyperparameters and loss coefficients.
    """

    name: str
    model: AutoencoderModelConfig
    architecture: AutoencoderArchitectureConfig
    train: AutoencoderTrainingConfig


@dataclass(frozen=True)
class WEncoderArchitectureConfig:
    """Architecture options for the codeword VAE.

    Attributes:
        encoder_h_dims: Hidden dimensions for the codeword encoder.
        decoder_h_dims: Hidden dimensions for the codeword decoder.
        dropout: Dropout probability of the codeword encoder and decoder.
    """

    encoder_h_dims: tuple[int, ...]
    decoder_h_dims: tuple[int, ...]
    dropout: UnitIntervalFloat


@dataclass(frozen=True)
class WEncoderModelConfig:
    """Model options for the codeword VAE.

    Attributes:
        z_dim: Continuous latent space dim.
        n_pseudo_inputs: Num of pseudo inputs.
    """

    z_dim: StrictlyPositiveInt
    n_pseudo_inputs: StrictlyPositiveInt


@dataclass(frozen=True)
class WEncoderLossConfig:
    """Specification for codeword VAE loss coefficients.

    Attributes:
        c_kld: Kullback-Leibler Divergence coefficient.
    """

    c_kld: PositiveFloat


@dataclass(frozen=True)
class WEncoderTrainingConfig:
    """Specification for codeword VAE training.

    Attributes:
        batch_size: Batch size for the vae.
        n_epochs: Number of total training epochs.
        loss: Loss configuration and coefficients.
        learn: Optimiser and scheduler configuration.
        pruning_window: Number of last validations the pruner averages when tuning.
    """

    batch_size: StrictlyPositiveInt
    n_epochs: StrictlyPositiveInt
    loss: WEncoderLossConfig
    learn: LearningConfig
    pruning_window: StrictlyPositiveInt


@dataclass(frozen=True)
class WAutoencodingConfig(PhaseConfig):
    """Specification for the codeword VAE phase.

    Attributes:
        name: Name of the phase.
        architecture: Architecture options.
        model: Model options.
        train: Training hyperparameters and loss coefficients.
    """

    name: str
    architecture: WEncoderArchitectureConfig
    model: WEncoderModelConfig
    train: WEncoderTrainingConfig


# ---------------------------------------------------------------------------
# Data configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset identity and class filter.

    Attributes:
        name: Dataset.
        select_classes: Select specific classes of the dataset ShapenetFlow.
        input_points: (Maximum) points of the training dataset.
        resample: Two different samplings for input and reference.
    """

    name: Datasets
    select_classes: list[ShapenetFlowClasses]
    input_points: InputPointsInt
    resample: bool


@dataclass(frozen=True)
class DataConfig:
    """Full data configuration.

    Attributes:
        dataset: Dataset selection and class parameters.
    """

    dataset: DatasetConfig


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for sample generation metrics evaluation.

    Attributes:
        n_chamfer_repeats: Number of tests for generation metrics based on Chamfer distance.
        n_emd_repeats: Number of tests for generation metrics based on Earth Mover distance.
    """

    n_chamfer_repeats: PositiveInt
    n_emd_repeats: PositiveInt


# ---------------------------------------------------------------------------
# User / environment settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlottingSettings:
    """Options for reconstruction rendering and visual inspection.

    Attributes:
        sample_indices: Sample indices to visualise.
        overlay: Overlay rendering mode (none, sampling_loop, graph_filter, components).
        show: Whether to also display the figures: the clouds and the latent space.
    """

    sample_indices: list[int]
    overlay: Overlays
    show: bool
    reconstruction_every: int


@dataclass(frozen=True)
class GenerationSettings:
    """Options for random sample generation.

    Attributes:
        n_generated: Number of generated samples.
        bias_dim: Latent dimension to bias during random generation.
        bias_value: Value of the bias in the latent dimension.
    """

    n_generated: PositiveInt
    bias_dim: PositiveInt
    bias_value: float


@dataclass(frozen=True)
class TrackerList:
    """Flags controlling which experiment trackers are active.

    Attributes:
        hydra: Saves a copy of the resolved config to the Hydra output dir.
        csv: Appends metrics to a CSV file in the experiment directory.
        tensorboard: Streams metrics to a local TensorBoard server.
        wandb: Uploads metrics to Weights & Biases (requires ``track_all`` extra).
        sqlalchemy: Persists metrics in a SQLite database (requires ``track_all`` extra).
    """

    hydra: bool
    csv: bool
    tensorboard: bool
    wandb: bool
    sqlalchemy: bool
    pyvista: bool
    plotly: bool


@dataclasses.dataclass(frozen=True)
class HydraSettings:
    """Subset of Hydra runtime settings, populated automatically at startup.

    Attributes:
        output_dir: Directory where Hydra stores logs and config dumps.
        job_logging: Hydra job-logging DictConfig used to configure Python logging.
        config_name: Name of the configuration file the settings were composed from.
    """

    output_dir: pathlib.Path = pathlib.Path('outputs')
    job_logging: Any = None
    config_name: str = ''


@dataclass(frozen=True)
class PathSpecs:
    """Filesystem path specifications populated from environment variables.

    Attributes:
        data_dir: Root directory holding the datasets, shared by all versions.
        metadata_dir: Directory holding the class names of the datasets.
        root_exp_dir: Root directory under which all experiment artefacts are stored.

    Note:
        Paths are loaded from the ``.env`` file via :class:`~src.config.environment.EnvSettings`.
    """

    _env = EnvSettings()
    data_dir: pathlib.Path = _env.data_dir
    metadata_dir: pathlib.Path = ROOT_DIR / 'metadata'
    root_exp_dir: pathlib.Path = _env.root_exp_dir

    @property
    def version_dir(self) -> pathlib.Path:
        """Version-scoped experiment directory: ``<root_exp_dir>/v<VERSION>``."""
        return self.root_exp_dir / f'v{VERSION}'


@dataclass(frozen=True)
class UserSettings:
    """User-specific runtime preferences.

    Attributes:
        device: Device where the models run: ``auto`` for the accelerator when available, or a name such as
            ``cpu`` or ``cuda:1``.
        n_workers: DataLoader worker processes. ``0`` runs loading in the main process.
        n_subprocesses: Number of DDP processes. ``0`` disables distributed training.
        trackers: Active experiment-tracking backends.
        plot: Reconstruction rendering and visual inspection options.
        generation: Random sample generation options.
        seed: Global random seed for reproducibility. ``None`` leaves it unset.
        torch_compile: Whether to compile the model with torch.compile.
        load: Checkpoint epoch to resume from: ``0`` skip; ``-1`` loads last checkpoint.
        run_id: Run to resume; empty resumes the last run of the experiment.
    """

    device: Device
    n_workers: PositiveInt
    n_subprocesses: PositiveInt
    trackers: TrackerList
    plot: PlottingSettings
    generation: GenerationSettings
    seed: int | None
    torch_compile: bool
    load: int
    run_id: str | None
    hydra: HydraSettings = dataclasses.field(default_factory=HydraSettings)
    path: PathSpecs = dataclasses.field(default_factory=PathSpecs)

    def __post_init__(self) -> None:
        """Apply the random seed immediately after the config is constructed."""
        if self.seed is not None:
            set_seed(self.seed)

        return


# ---------------------------------------------------------------------------
# Root config — main entry point for the configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration object for all experiment settings.

    Attributes:
        autoencoding: Configuration for the autoencoder training phase.
        data: Data pre-processing and dataset configuration.
        user: User-specific options and runtime preferences.
        w_autoencoding: Configuration for the codeword VAE training phase, absent for the plain autoencoder.
        evaluation: Settings for sample generation and evaluation.
        final: Uses val dataset for training and test dataset for testing, otherwise test on val.
        variation: Variation name for the experiment (defaults to git branch if None).
        tags: List of tags describing active configuration overrides.
    """

    autoencoding: AutoencodingConfig
    data: DataConfig
    user: UserSettings
    w_autoencoding: WAutoencodingConfig | None
    evaluation: EvaluationConfig
    final: bool
    variation: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    version: str = dataclasses.field(default=f'v{VERSION}', init=False)

    @cached_property
    def git_branch(self) -> str:
        """Current git branch (used when variation is unset)."""
        try:
            return get_current_branch()
        except ValueError as ve:
            msg = "Could not resolve git branch. Set 'variation' in defaults.yaml."
            raise ValueError(msg) from ve

    @property
    def resolved_variation(self) -> str:
        """Variation if set, otherwise the current git branch."""
        return self.variation or self.git_branch

    @property
    def name(self) -> str:
        """Full experiment name embedded into tracker runs."""
        suffix = '_final' if self.final else ''
        return self.resolved_variation[:249] + suffix

    @property
    def par_dir(self) -> pathlib.Path:
        """Directory of the experiments of this configuration: ``<version_dir>/<config_name>``."""
        return self.user.path.version_dir / self.user.hydra.config_name

    @property
    def project(self) -> str:
        """Tracker project name, one per version."""
        return f'{PROJECT_NAME}-{self.version}'
