"""Best reachable IoU, the same-size grid, and the automatic verdict."""
from argparse import Namespace
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import evaluate_partitions as ep


def exhaustive_best_iou(gt, partition):
    ids = [value for value in np.unique(partition) if value != 0]
    best = 0.
    for size in range(1, len(ids) + 1):
        for subset in combinations(ids, size):
            selected = np.isin(partition, subset)
            best = max(best, (selected & gt).sum() / (selected | gt).sum())
    return best


@pytest.mark.parametrize('seed', range(20))
def test_best_iou_matches_trying_every_union_of_cells(seed):
    rng = np.random.default_rng(seed)
    partition = rng.integers(0, 8, (24, 24))
    gt = rng.random((24, 24)) < rng.uniform(.1, .7)
    assert ep.maximize_iou(gt, partition)[0] == pytest.approx(exhaustive_best_iou(gt, partition))


def test_exact_cells_give_iou_one_and_full_coverage():
    partition = np.array([[1, 1, 2, 2]] * 4)
    assert ep.maximize_iou(partition == 1, partition) == pytest.approx((1, 100))


def test_background_cell_is_never_selected():
    partition = np.array([[0, 0, 2, 2]] * 4)
    iou, coverage = ep.maximize_iou(partition == 0, partition)
    assert (iou, coverage) == (0, 0)


def test_partial_coverage_is_reported_in_percent():
    partition = np.array([[1, 1, 0, 0]] * 4)
    iou, coverage = ep.maximize_iou(np.ones((4, 4), dtype=bool), partition)
    assert iou == pytest.approx(.5)
    assert coverage == pytest.approx(50)


def test_empty_gt_scores_zero():
    assert ep.maximize_iou(np.zeros((4, 4), dtype=bool), np.ones((4, 4), dtype=int)) == (0, 0)


@pytest.mark.parametrize('shape, cells', [((480, 640), 60), ((100, 100), 1), ((50, 300), 17), ((7, 5), 500)])
def test_grid_covers_every_pixel_with_about_the_requested_cells(shape, cells):
    grid = ep.grid_partition(shape, cells)
    assert grid.shape == shape and grid.min() == 1
    assert len(np.unique(grid)) >= min(cells, shape[0] * shape[1])
    assert len(np.unique(grid)) <= max(cells * 1.5, 1)


def settings():
    return Namespace(min_images=30, use_median_iou=.75, use_iou_50=.75, skip_median_iou=.5, min_gain=.05)


def summary(**changes):
    return {'found': 300, 'images': 300, 'median_iou': .82, 'gain': .2, 'iou_50': .79, **changes}


@pytest.mark.parametrize('changes, expected', [
    ({}, ep.USE),
    ({'found': 2, 'images': 2}, ep.NEED_PARTITIONS),
    ({'images': 5}, ep.NO_GT),
    ({'median_iou': .6}, ep.CAUTION),
    ({'iou_50': .6}, ep.CAUTION),
    ({'gain': .01}, ep.CAUTION),
    ({'median_iou': .4}, ep.SKIP),
    ({'gain': -.01}, ep.SKIP),
])
def test_verdict(changes, expected):
    assert ep.decide(summary(**changes), settings())[0] == expected


def test_notebook_coco_reference_counts_as_use():
    verdict, _ = ep.decide(summary(median_iou=.825, iou_50=.793), settings())
    assert verdict == ep.USE


def test_a_broken_dataset_does_not_stop_the_others(tmp_path, monkeypatch, capsys):
    def evaluate(name, args):
        if name == 'bad':
            raise FileNotFoundError('missing annotations')
        return [(ep.summarize(name, 'folder', 40, 40, [], args), [])]
    monkeypatch.setattr(ep, 'evaluate_dataset', evaluate)
    args = ep.parser().parse_args(['--config', 'bad', 'good', '--output-dir', str(tmp_path)])
    verdicts = {item['config']: item['verdict'] for item in ep.run(args)}
    assert verdicts['bad'] == ep.ERROR
    assert verdicts['good'] == ep.SKIP
    assert (tmp_path / 'conclusion.md').is_file()


def test_each_gt_label_is_scored_separately_and_ignored_pixels_are_dropped(monkeypatch):
    labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1000, 1000, 0, 0], [0, 0, 0, 0]])
    monkeypatch.setattr(ep, 'load_ground_truth', lambda *_: {
        'mask': np.isin(labels, [1, 2]), 'label_map': labels, 'label_names': {1: 'wing'}})
    objects = list(ep.gt_objects({'data': {'name': 'CAMO'}}, Path('a.jpg'), labels.shape))
    assert [name for name, _, _ in objects] == ['wing', '2']
    assert objects[0][1].sum() == 4
    assert not objects[0][2][2, 0]


def test_all_sam_finds_each_partition_folder_of_the_dataset(tmp_path, monkeypatch):
    for name in ['CAMO_sam1_vit_b_62', 'CAMO_sam2_tiny_62', 'BUSI_sam1_vit_b_62']:
        (tmp_path / name).mkdir()
    (tmp_path / 'CAMO_sam1_vit_b_62.log').write_text('')
    monkeypatch.setattr(ep, 'partition_path', lambda _: tmp_path / 'CAMO_sam2_tiny_62')
    assert [path.name for path in ep.partition_folders({}, True)] == ['CAMO_sam1_vit_b_62', 'CAMO_sam2_tiny_62']
    assert [path.name for path in ep.partition_folders({}, False)] == ['CAMO_sam2_tiny_62']
