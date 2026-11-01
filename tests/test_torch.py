"""Tests for torch config utilities."""

import pytest
import torch

from src.config.torch import get_mixed_precision


@pytest.mark.parametrize(
    ('precision', 'expected'),
    [
        ('float16', torch.float16),
        ('bfloat16', torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_get_mixed_precision(precision: str | torch.dtype, expected: torch.dtype) -> None:
    """Resolve the autocast precision by name or as a dtype."""
    dtype = get_mixed_precision(precision)

    assert dtype is expected


def test_get_mixed_precision_none() -> None:
    """Keep None, which trains in full precision."""
    dtype = get_mixed_precision(None)

    assert dtype is None


@pytest.mark.parametrize('precision', ['float32', torch.float32, 'half', 'fp16'])
def test_get_mixed_precision_raises(precision: str | torch.dtype) -> None:
    """Raise ValueError for a precision other than float16 and bfloat16."""
    with pytest.raises(ValueError, match='not float16 or bfloat16'):
        get_mixed_precision(precision)
