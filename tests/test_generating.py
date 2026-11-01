"""Tests for the scores comparing a generated set with a reference set."""

import dataclasses

import pytest
import pytest_mock
import torch

from src.utils.distances import torch_chamfer
from src.data.preprocessing import normalize_clouds
from src.utils.generation import (
    GenerativeScores,
    get_generative_scores,
    get_pairwise_distances,
)

CPU = torch.device('cpu')


@dataclasses.dataclass
class _Model:
    name: str = 'vqvae_vae'
    epoch: int = 7


def _cloud(offset: float) -> torch.Tensor:
    return torch.tensor([[offset, 0.0, 0.0], [offset + 0.1, 0.0, 0.0]])


class TestGenerativeScores:
    """Test the scores of generated clouds against reference clouds."""

    @pytest.fixture(autouse=True)
    def setup(self, mocker: pytest_mock.MockerFixture) -> None:
        """Set up the reference clouds and record the published events."""
        self.reference = [_cloud(offset) for offset in (0.0, 1.0, 2.0, 3.0)]
        self.mock_event = mocker.patch('src.utils.generation.log_events.MetricEvent')
        return

    @pytest.fixture
    def scores(self) -> GenerativeScores:
        """Provide the scores with the Chamfer distance."""
        return GenerativeScores(self.reference, torch_chamfer, name='Generation_Chamfer', device=CPU)

    def test_a_copy_of_the_reference_set_scores_perfectly(self, scores: GenerativeScores) -> None:
        """Every cloud has its copy in the other set as nearest neighbour."""
        generated = [cloud.clone() for cloud in self.reference]

        result = scores(_Model(), generated)

        assert result == {'MMD': pytest.approx(0.0), 'Coverage': 1.0, '1-NNA': 0.0}

    def test_scores_are_published_for_the_model(self, scores: GenerativeScores) -> None:
        """The scores are published with the name and the epoch of the model."""
        generated = [_cloud(offset) for offset in (0.5, 1.5, 2.5, 3.5)]

        result = scores(_Model(), generated)

        self.mock_event.assert_called_once_with(
            model_name='vqvae_vae', source_name='Generation_Chamfer', epoch=7, metrics=result
        )


def test_get_generative_scores() -> None:
    """The scores follow their definitions on a hand-computed example.

    Nearest neighbours: reference 0 -> generated 0 (other set), reference 1 ->
    reference 0 (same set), generated 0 -> reference 0 (other set), generated 1
    -> generated 0 (same set).
    """
    reference_distances = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cross_distances = torch.tensor([[0.1, 5.0], [4.0, 3.0]])
    generated_distances = torch.tensor([[0.0, 2.0], [2.0, 0.0]])

    result = get_generative_scores(reference_distances, cross_distances, generated_distances)

    # closest generated: 0.1 for reference 0 and 3 for reference 1
    assert result['MMD'] == pytest.approx(1.55)

    # generated 0 is closest to reference 0, generated 1 to reference 1
    assert result['Coverage'] == 1.0
    assert result['1-NNA'] == 0.5


def test_get_generative_scores_of_distant_sets() -> None:
    """Every cloud is closest to its own set, and a collapsed set covers one cloud."""
    close = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    far = torch.full((3, 3), 10.0)
    far[0, :] = 9.0

    result = get_generative_scores(close, far, close)

    assert result == {'MMD': pytest.approx(29 / 3), 'Coverage': pytest.approx(1 / 3), '1-NNA': 1.0}


def test_get_generative_scores_needs_two_clouds_in_each_set() -> None:
    """A set of one cloud has no nearest neighbour inside itself."""
    with pytest.raises(ValueError, match='at least two clouds'):
        get_generative_scores(torch.zeros(2, 2), torch.zeros(2, 1), torch.zeros(1, 1))


def test_get_pairwise_distances() -> None:
    """Each entry is the Chamfer distance of the pair, averaged over points."""
    first = [torch.randn(5, 3) for _ in range(3)]
    second = [torch.randn(5, 3) for _ in range(40)]

    distances = get_pairwise_distances(first, second, torch_chamfer, device=CPU)

    expected = torch.tensor([[float(torch_chamfer(a[None], b[None])) / 5 for b in second] for a in first])
    assert distances.shape == (3, 40)
    torch.testing.assert_close(distances, expected)


def test_normalize_clouds() -> None:
    """Normalized clouds are centered with radius one, and the input is untouched."""
    clouds = torch.randn(3, 20, 3) * 5 + 2
    original = clouds.clone()

    normalized = normalize_clouds(clouds)

    assert torch.equal(clouds, original)
    torch.testing.assert_close(normalized.mean(dim=1), torch.zeros(3, 3), atol=1e-5, rtol=0)
    torch.testing.assert_close(normalized.norm(dim=2).max(dim=1)[0], torch.ones(3))
