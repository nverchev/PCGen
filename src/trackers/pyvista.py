"""PyVista tracker for 3D point cloud visualisation and rendering."""

import enum
import functools
import pathlib

from collections.abc import Sequence
from typing import ClassVar, Final, cast, override

import numpy as np
import pyvista as pv
import torch

from drytorch.core import log_events
from drytorch.trackers.base_classes import Dumper

from src.lib.events import GenerationEvent, ReconstructionEvent
from src.config.options import Colorscale, Overlays
from src.data.structures import Inputs, Outputs
from src.utils.neighbour_ops import graph_filtering

__all__ = [
    'BLUE',
    'COLOR_SEQUENCE',
    'GREEN',
    'ORANGE',
    'RED',
    'VIOLET',
    'Modes',
    'PyVistaTracker',
]

BLUE = np.array([0.3, 0.3, 0.9])
RED = np.array([0.9, 0.3, 0.3])
GREEN = np.array([0.3, 0.9, 0.3])
VIOLET = np.array([0.6, 0.0, 0.9])
ORANGE = np.array([0.9, 0.6, 0.0])
COLOR_SEQUENCE = [BLUE, RED, GREEN, VIOLET, ORANGE]


class Modes(enum.StrEnum):
    """What the rendered clouds come from."""

    gen = 'gen'
    recon = 'recon'
    double_recon = 'double_recon'
    original = 'original'


class PyVistaTracker(Dumper):
    """PyVista tracker for rendering point clouds, reconstructions, and generations.

    Attributes:
        show: Whether a window also displays the clouds.
        overlay: Overlay rendering mode.
        sample_indices: Sample indices to visualise.
        class_names: Names of the classes, added to the names of the images.
        m_test: Points of each cloud, before any sampling loop.
        n_components: Components of the decoder.
        save_dir: Directory where screenshots are saved, if not the one of the run.
    """

    folder_name: ClassVar[str] = 'pyvista'
    show: Final[bool]
    overlay: Final[Overlays]
    sample_indices: Final[Sequence[int]]
    class_names: Final[Sequence[str]]
    m_test: Final[int]
    n_components: Final[int]
    save_dir: Final[pathlib.Path | None]

    def __init__(
        self,
        *,
        show: bool,
        overlay: Overlays,
        sample_indices: Sequence[int],
        class_names: Sequence[str],
        m_test: int,
        n_components: int,
        par_dir: pathlib.Path | None = None,
        save_dir: pathlib.Path | None = None,
    ) -> None:
        """Initialize tracker.

        Args:
            show: whether a window also displays the clouds.
            overlay: overlay rendering mode.
            sample_indices: sample indices to visualise.
            class_names: names of the classes, added to the names of the images.
            m_test: points of each cloud, before any sampling loop.
            n_components: components of the decoder.
            par_dir: parent directory for tracker data when using Dumper.
            save_dir: optional explicit directory where screenshots are saved.
        """
        super().__init__(par_dir=par_dir)
        self.show = show
        self.overlay = overlay
        self.sample_indices = sample_indices
        self.class_names = class_names
        self.m_test = m_test
        self.n_components = n_components
        self.save_dir = save_dir
        return

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: ReconstructionEvent) -> None:
        if not (self.show or self.record):
            return super().notify(event)

        mode = Modes.double_recon if event.is_double else Modes.recon
        self._render_outputs(
            event.outputs,
            mode=mode,
            save_dir=self._resolve_save_dir() if self.record else None,
            sample_indices=event.sample_indices,
            epoch=event.epoch,
            inputs=event.inputs,
        )
        return super().notify(event)

    @notify.register
    def _(self, event: GenerationEvent) -> None:
        if not (self.show or self.record):
            return super().notify(event)

        self._render_outputs(
            event.outputs,
            mode=Modes.gen,
            save_dir=self._resolve_save_dir() if self.record else None,
            epoch=event.epoch,
        )
        return super().notify(event)

    def _get_sample_index(
        self,
        index: int,
        mode: Modes,
        sample_indices: Sequence[int] | None = None,
    ) -> int:
        if mode in (Modes.recon, Modes.double_recon, Modes.original):
            if sample_indices is not None and index < len(sample_indices):
                return sample_indices[index]

            if index < len(self.sample_indices):
                return self.sample_indices[index]

        return index

    def _get_name(
        self,
        index: int,
        mode: Modes,
        overlay: str | None = None,
        sample_indices: Sequence[int] | None = None,
        epoch: int | None = None,
    ) -> str:
        sample_idx = self._get_sample_index(index, mode, sample_indices=sample_indices)
        parts = [str(mode), *self.class_names]
        if overlay is not None:
            parts.append(overlay)

        if epoch is not None:
            parts.append(f'epoch_{epoch}')

        parts.append(str(sample_idx))
        return '_'.join(parts)

    def _render_components(
        self,
        cloud: torch.Tensor,
        outputs: Outputs,
        *,
        index: int,
        mode: Modes,
        save_dir: pathlib.Path | None = None,
        sample_indices: Sequence[int] | None = None,
        epoch: int | None = None,
    ) -> None:
        if not outputs.attention.numel():
            msg = 'The configured decoder mixes one component, which has nothing to show.'
            raise ValueError(msg)

        strength, claimed = outputs.attention[index].max(dim=1)
        groups = (claimed.cpu() + 1) * (strength > 0.0).cpu()
        by_group = [cloud[groups == group] for group in range(self.n_components + 1)]
        self.render_cloud(
            by_group,
            name=self._get_name(index, mode, Overlays.COMPONENTS, sample_indices=sample_indices, epoch=epoch),
            show=self.show,
            save_dir=save_dir,
        )
        spread: list[torch.Tensor] = [torch.empty(0)]
        for position, component in enumerate(outputs.components[index].cpu().transpose(1, 0).unbind(2)):
            shift = torch.tensor([[(1 - self.n_components) / 2 + position, 0.0, 0.0]])
            spread.append((component + shift) / self.n_components)

        self.render_cloud(
            spread,
            name=self._get_name(index, mode, 'spread', sample_indices=sample_indices, epoch=epoch),
            show=self.show,
            save_dir=save_dir,
        )
        return

    def _render_outputs(
        self,
        outputs: Outputs,
        *,
        mode: Modes,
        save_dir: pathlib.Path | None = None,
        sample_indices: Sequence[int] | None = None,
        epoch: int | None = None,
        inputs: Inputs | None = None,
    ) -> None:
        original_clouds = None
        if inputs is not None and inputs.cloud.numel() > 0:
            original_clouds = inputs.cloud.detach().cpu()

        clouds, *loops = outputs.recon.cpu().split(self.m_test, dim=1)
        for index, cloud in enumerate(clouds):
            sample_idx = self._get_sample_index(index, mode, sample_indices=sample_indices)
            sample_dir = None if save_dir is None else save_dir / str(sample_idx)

            if original_clouds is not None and index < len(original_clouds):
                orig_cloud = original_clouds[index]
                orig_name = self._get_name(index, Modes.original, sample_indices=sample_indices)
                self.render_cloud(
                    [orig_cloud],
                    name=orig_name,
                    show=self.show,
                    save_dir=sample_dir,
                )

            if self.overlay == Overlays.SAMPLING_LOOP:
                name = self._get_name(index, mode, Overlays.SAMPLING_LOOP, sample_indices=sample_indices, epoch=epoch)
                self.render_cloud([cloud, loops[0][index]], name=name, show=self.show, save_dir=sample_dir)
            elif self.overlay == Overlays.COMPONENTS:
                self._render_components(
                    cloud,
                    outputs,
                    index=index,
                    mode=mode,
                    save_dir=sample_dir,
                    sample_indices=sample_indices,
                    epoch=epoch,
                )
            elif self.overlay == Overlays.GRAPH_FILTER:
                filtered = graph_filtering(cloud.transpose(0, 1).unsqueeze(0)).squeeze().transpose(0, 1)
                name = self._get_name(index, mode, Overlays.GRAPH_FILTER, sample_indices=sample_indices, epoch=epoch)
                self.render_cloud(
                    [cloud],
                    name=name,
                    arrows=filtered - cloud,
                    show=self.show,
                    save_dir=sample_dir,
                )
            else:
                name = self._get_name(index, mode, sample_indices=sample_indices, epoch=epoch)
                self.render_cloud([cloud], name=name, show=self.show, save_dir=sample_dir)

        return

    def _resolve_save_dir(self) -> pathlib.Path:
        """Resolve output directory for rendered screenshots."""
        if self.save_dir is not None:
            return self.save_dir

        try:
            return self._get_run_dir()
        except Exception:
            return pathlib.Path(self.folder_name)

    @classmethod
    def render_cloud(
        cls,
        clouds: Sequence[torch.Tensor] | torch.Tensor,
        *,
        name: str,
        colorscale: Colorscale = Colorscale.SEQUENCE,
        show: bool = True,
        arrows: torch.Tensor | None = None,
        save_dir: pathlib.Path | None = None,
    ) -> None:
        """Render 3D point cloud(s) with PyVista.

        Args:
            clouds: A sequence of point clouds or a single cloud with shape (N, 3).
            name: Name of the output image file.
            colorscale: Colorscale enum to use for colouring the clouds.
            show: Whether a window displays the clouds.
            arrows: Optional vector field tensor with shape (N, 3) to render as glyph arrows.
            save_dir: Directory where the image is saved; ``None`` saves nothing.
        """
        if isinstance(clouds, torch.Tensor):
            clouds = [clouds]

        plotter = pv.Plotter(
            lighting='three lights',
            window_size=[1024, 1024],
            notebook=False,
            off_screen=not show,
        )
        plotter.camera_position = pv.CameraPosition((-3.0, -2.1, 1.1), focal_point=(0, 0, 0), viewup=(0, 0, 1))
        for i in [-1, 1]:
            light_point = (i, 1, 0)
            light = pv.Light(position=light_point, focal_point=(0, 0, 0), intensity=0.2, positional=True)
            plotter.add_light(light)

        for i, cloud_tensor in enumerate(clouds):
            if len(cloud_tensor) == 0:
                continue

            if colorscale == Colorscale.BLUERED:
                color = (1 - i) * BLUE + i * RED
            elif colorscale == Colorscale.SEQUENCE:
                color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]
            else:
                msg = f'Colorscale {colorscale} not available'
                raise ValueError(msg)

            n = cloud_tensor.shape[0]
            poly = pv.PolyData(cloud_tensor[:, [0, 2, 1]].detach().cpu().numpy())
            geom = pv.Sphere(theta_resolution=8, phi_resolution=8)
            poly['radius'] = 0.05 * np.ones(n)
            glyphs = cast(pv.PolyData, poly.glyph(scale='radius', geom=geom, orient=False))
            plotter.add_mesh(glyphs, color=color, point_size=15, render_points_as_spheres=True, smooth_shading=True)
            if arrows is not None:
                arrow_geom = pv.Arrow(shaft_radius=0.1, tip_radius=0.2, scale=1)
                poly['vectors'] = arrows[:, [0, 2, 1]].detach().cpu().numpy()
                poly.set_active_vectors('vectors')
                arrows_glyph = cast(pv.PolyData, poly.glyph(orient='vectors', geom=arrow_geom))
                plotter.add_mesh(
                    arrows_glyph,
                    lighting=True,
                    line_width=10,
                    color=RED,
                    show_scalar_bar=False,
                    edge_color=RED,
                )

        if show:
            plotter.background_color = pv.Color('white')
            plotter.show(auto_close=False)  # keeps the plotter for the screenshot

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = name if name.endswith('.png') else f'{name}.png'
            plotter.screenshot(str(save_dir / filename), window_size=[1024, 1024], transparent_background=True)

        plotter.close()
        return
