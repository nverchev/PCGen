"""Defines all enumerated options used in configuration files."""

import enum

__all__ = [
    'Colorscale',
    'Datasets',
    'Decoders',
    'Encoders',
    'ModelHeads',
    'Overlays',
    'ReconLosses',
    'Schedulers',
    'ShapenetFlowClasses',
]


# ---------------------------------------------------------------------------
# Project Specific: Choices for PCGen
# ---------------------------------------------------------------------------


class Datasets(enum.StrEnum):
    """Dataset choices."""

    SHAPENET_ATLAS = enum.auto()
    SHAPENET_FLOW = enum.auto()
    COINS = enum.auto()


class ShapenetFlowClasses(enum.StrEnum):
    """Select specific classes of the dataset ShapenetFlow."""

    AIRPLANE = enum.auto()
    CAR = enum.auto()
    CHAIR = enum.auto()


class ModelHeads(enum.StrEnum):
    """Model head choices."""

    AE = enum.auto()
    VQVAE = enum.auto()
    ORACLE = enum.auto()


class Encoders(enum.StrEnum):
    """Encoder choices."""

    LDGCNN = enum.auto()
    DGCNN = enum.auto()
    FOLDINGNET = enum.auto()


class Decoders(enum.StrEnum):
    """Decoder choices."""

    PCGEN = enum.auto()
    PCGEN_CONCAT = enum.auto()
    FULL = enum.auto()
    FOLDINGNET = enum.auto()
    TEARINGNET = enum.auto()
    ATLASNET_DEFORMATION = enum.auto()
    ATLASNET_TRANSLATION = enum.auto()


class ReconLosses(enum.StrEnum):
    """Reconstruction loss choices."""

    CHAMFER = enum.auto()
    CHAMFER_EMD = enum.auto()


class Overlays(enum.StrEnum):
    """Visualisation overlay choices."""

    NONE = enum.auto()
    SAMPLING_LOOP = enum.auto()
    GRAPH_FILTER = enum.auto()
    COMPONENTS = enum.auto()


class Colorscale(enum.StrEnum):
    """Colour scale options for point cloud rendering."""

    SEQUENCE = enum.auto()
    BLUERED = enum.auto()


# ---------------------------------------------------------------------------
# Generic options available in drytorch
# ---------------------------------------------------------------------------


class Schedulers(enum.StrEnum):
    """Learning-rate scheduler names."""

    CONSTANT = enum.auto()
    COSINE = enum.auto()
    EXPONENTIAL = enum.auto()
