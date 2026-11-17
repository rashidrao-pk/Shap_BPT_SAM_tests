"""Partition previews: label 0 is black, every other label a distinct, non-black colour."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from partition_colors import colorize  # noqa: E402


def test_background_is_black_and_regions_are_distinct():
    labels = np.repeat(np.arange(40)[None, :], 2, axis=0)
    colored = colorize(labels)
    assert np.all(colored[labels == 0] == 0)
    regions = colored[0, 1:]
    assert regions.max(axis=1).min() > 0.2  # No region is near black.
    assert len(np.unique(regions.round(6), axis=0)) == 39


def test_partition_without_label_zero_has_no_black():
    colored = colorize(np.array([[1, 2], [3, 4]]))
    assert colored.max(axis=-1).min() > 0.2
