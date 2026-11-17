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


def test_count_title_shows_sam_to_compact():
    from partition_colors import count_title, region_count
    assert region_count(np.array([[0, 3], [5, 5]])) == 2
    assert count_title(19, 62) == 'SAM → Compact\n62 → 19 regions'
    assert count_title(19) == 'Compact\n19 regions'


def test_sam_region_count_reads_the_sibling_sam_map(tmp_path):
    import evaluate_partitions
    np.save(tmp_path / '139_sam.npy', np.arange(12).reshape(3, 4))
    np.save(tmp_path / '139_compact.npy', np.zeros((3, 4), dtype=np.int64))
    assert evaluate_partitions.sam_region_count(tmp_path / '139_compact.npy', (3, 4), 'compact') == 11
    assert evaluate_partitions.sam_region_count(tmp_path / '7_compact.npy', (3, 4), 'compact') is None
    assert evaluate_partitions.sam_region_count(tmp_path / '139_sam.npy', (3, 4), 'sam') is None


def test_gt_object_names_are_written_inside_their_objects():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from partition_quality_plots import label_objects
    gt = np.zeros((40, 60), dtype=np.int64)
    gt[5:35, 5:25], gt[10:30, 35:55] = 1, 2
    figure, ax = plt.subplots()
    label_objects(ax, gt, ['person', 'dog'])
    placed = {text.get_text(): text.get_position() for text in ax.texts}
    plt.close(figure)
    assert set(placed) == {'person', 'dog'}
    for number, name in enumerate(['person', 'dog'], 1):
        column, row = placed[name]
        assert gt[int(row), int(column)] == number
