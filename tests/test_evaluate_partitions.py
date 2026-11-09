"""Best reachable IoU, the same-size grid, and the automatic verdict."""
from argparse import Namespace
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import evaluate_partitions as ep
import partition_diagnosis as diagnosis
import partition_quality_plots as plots


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
    assert ep.maximize_iou(partition == 1, partition) == (1, 100, [1])


def test_background_cell_is_never_selected():
    partition = np.array([[0, 0, 2, 2]] * 4)
    assert ep.maximize_iou(partition == 0, partition) == (0, 0, [])


def test_partial_coverage_is_reported_in_percent():
    partition = np.array([[1, 1, 0, 0]] * 4)
    iou, coverage, _ = ep.maximize_iou(np.ones((4, 4), dtype=bool), partition)
    assert iou == pytest.approx(.5)
    assert coverage == pytest.approx(50)


def test_empty_gt_scores_zero():
    assert ep.maximize_iou(np.zeros((4, 4), dtype=bool), np.ones((4, 4), dtype=int)) == (0, 0, [])


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


def object_row(config, image_id, iou, share, cause, name='cat'):
    return {'config': config, 'image_id': image_id, 'object': name, 'best_iou': iou, 'grid_iou': .2, 'gt_coverage': iou * 100, 'gt_share': share,
            'size': diagnosis.size_band(share), 'cells': 10, 'cause': cause}


def test_notebook_coco_reference_counts_as_use():
    verdict, _ = ep.decide(summary(median_iou=.825, iou_50=.793), settings())
    assert verdict == ep.USE


def test_a_broken_dataset_does_not_stop_the_others(tmp_path, monkeypatch, capsys):
    def evaluate(name, args):
        if name == 'bad':
            raise FileNotFoundError('missing annotations')
        return [(ep.summarize(name, 'folder', 40, 40, [], args), [])]
    monkeypatch.setattr(ep, 'evaluate_dataset', evaluate)
    args = ep.parser().parse_args(['--config', 'bad', 'good', '--output-dir', str(tmp_path), '--no-plots'])
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


def test_the_default_uses_every_completed_partition():
    assert ep.parser().parse_args([]).limit is None


@pytest.mark.parametrize('seed', range(5))
def test_selected_cells_reproduce_the_reported_iou(seed):
    rng = np.random.default_rng(seed)
    partition = rng.integers(0, 12, (30, 30))
    gt = rng.random((30, 30)) < .4
    iou, coverage, cells = ep.maximize_iou(gt, partition)
    selected = np.isin(partition, cells)
    assert iou == pytest.approx((selected & gt).sum() / (selected | gt).sum())
    assert coverage == pytest.approx((selected & gt).sum() / gt.sum() * 100)


def test_examples_are_the_highest_middle_and_lowest_images():
    rows = [{'image_id': image, 'best_iou': iou} for image, iou in
            [('a', .9), ('a', .7), ('b', .1), ('c', .5), ('d', .6), ('e', .2)]]
    assert ep.example_ids(rows) == {'Highest IoU': 'a', 'Middle IoU': 'c', 'Lowest IoU': 'b'}


def test_case_maps_and_plots_on_a_small_dataset(tmp_path, monkeypatch):
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / 'a.png')
    partition = np.repeat(np.repeat(np.array([[1, 2], [3, 0]], dtype=np.uint16), 4, 0), 4, 1)
    np.save(tmp_path / 'a_compact.npy', partition)
    first, second = partition == 1, partition == 3
    monkeypatch.setattr(ep, 'gt_objects', lambda *_: [('x', first, None), ('y', second, None)])
    case = ep.case_maps({}, tmp_path / 'a.png', tmp_path / 'a_compact.npy')
    assert case['objects'] == 2 and case['cells'] == 3 and case['mean_iou'] == 1
    assert (case['subsets'] == first + 2 * second).all()
    assert (case['closest'] == case['gt']).all()

    plots.plot_examples({'Highest IoU': case}, 'Toy SAM1', tmp_path / 'examples')
    args = ep.parser().parse_args(['--output-dir', str(tmp_path / 'plots')])
    args.output_dir.mkdir()
    rows = [object_row(name, 'i', value, 5., 'Merged' if value < .5 else '') for name in 'AB' for value in (.2, .6, .9)]
    summaries = [{**ep.summarize(name, 'folder', 40, 40, [], args), 'verdict': ep.USE} for name in 'AB']
    diagnoses = {name: diagnosis.diagnose([row for row in rows if row['config'] == name]) for name in 'AB'}
    plots.plot_all([*summaries, {**summaries[0], 'config': 'C', 'verdict': ep.NEED_PARTITIONS}], rows, args, diagnoses)
    made = {path.name for path in args.output_dir.glob('*.png')}
    assert made == {'best_iou_boxplot.png', 'best_iou_violin.png', 'gt_coverage_boxplot.png', 'gt_coverage_violin.png',
                    'iou_thresholds.png', 'overview.png', 'failure_causes.png', 'iou_by_object_size.png'}
    assert (tmp_path / 'examples' / 'Toy_SAM1.png').is_file()


@pytest.mark.parametrize('share, expected', [(.05, 'Tiny (<0.1%)'), (.1, 'Small (0.1-1%)'), (5, 'Medium (1-10%)'), (40, 'Large (>10%)')])
def test_size_bands(share, expected):
    assert diagnosis.size_band(share) == expected


@pytest.mark.parametrize('iou, precision, unassigned, expected', [
    (.8, .9, 0, ''), (.2, .9, .7, 'Unassigned'), (.2, .1, 0, 'Merged'), (.3, .8, 0, 'Partial')])
def test_failure_cause(iou, precision, unassigned, expected):
    assert diagnosis.failure_cause(iou, precision, unassigned) == expected


def test_diagnosis_reports_causes_sizes_classes_and_worst_images():
    rows = [object_row('A', 'bad', .1, .05, 'Merged', 'kite') for _ in range(15)]
    rows += [object_row('A', 'good', .9, 20, '', 'bus') for _ in range(15)]
    result = diagnosis.diagnose(rows)
    assert result['low'] == 15 and result['causes']['Merged'] == 1
    assert result['groups'][0]['group'] == 'kite'
    assert result['images'][0]['image_id'] == 'bad' and result['images'][0]['main_cause'] == 'Merged'
    assert result['larger_median'] == pytest.approx(.9)
    text = diagnosis.report([{'config': 'A', 'verdict': ep.CAUTION, 'reason': 'low', 'median_iou': .5, 'iou_50': .5, 'gain': .3}], {'A': result}, 1)
    assert 'Merged' in text and 'kite' in text and '| bad |' in text


def test_a_two_point_polygon_is_dropped_instead_of_crashing():
    from pycocotools.coco import COCO
    coco = COCO()
    coco.dataset = {'images': [{'id': 1, 'height': 10, 'width': 10}], 'categories': [], 'annotations': []}
    coco.createIndex()
    square = [1, 1, 8, 1, 8, 8, 1, 8]
    assert ep.annotation_mask(coco, {'image_id': 1, 'segmentation': [[1, 1, 2, 2]]}) is None
    assert ep.annotation_mask(coco, {'image_id': 1, 'segmentation': [[1, 1, 2, 2], square]}).any()


def test_celeba_parts_are_combined_unless_split(monkeypatch):
    labels = np.array([[1, 1, 2, 2], [0, 0, 0, 0]])
    monkeypatch.setattr(ep, 'load_ground_truth', lambda *_: {'mask': labels > 0, 'label_map': labels, 'label_names': {}})
    config = {'data': {'name': 'CelebAHQ'}}
    assert [name for name, _, _ in ep.gt_objects(config, Path('a.jpg'), labels.shape, True)] == ['face']
    assert len(list(ep.gt_objects(config, Path('a.jpg'), labels.shape, False))) == 2
    assert ep.combines_parts(config, ep.parser().parse_args([]))
    assert not ep.combines_parts(config, ep.parser().parse_args(['--split-parts']))
    assert not ep.combines_parts({'data': {'name': 'CAMO'}}, ep.parser().parse_args([]))


def test_one_unreadable_image_does_not_lose_the_dataset(tmp_path, monkeypatch, capsys):
    calls = []

    def evaluate(config, label, image_id, *_):
        calls.append(image_id)
        if len(calls) == 1:
            raise TypeError('bad polygon')
        return [object_row(label, image_id, .9, 5., '')]
    for number in range(3):
        np.save(tmp_path / f'{number}_compact.npy', np.ones((2, 2), dtype=np.uint16))
    monkeypatch.setattr(ep, 'evaluate_image', evaluate)
    monkeypatch.setattr(ep, 'load_path_config', lambda _: {'data': {'name': 'CAMO'}})
    monkeypatch.setattr(ep, 'partition_path', lambda _: tmp_path)
    monkeypatch.setattr(ep, 'image_paths', lambda _: {str(number): Path(f'{number}.png') for number in range(3)})
    args = ep.parser().parse_args(['--no-plots', '--min-images', '1'])
    summary, rows = ep.evaluate_dataset('CAMO_epito', args)[0]
    assert (summary['failed'], summary['images'], len(rows)) == (1, 2, 2)
    assert 'could not be read' in summary['reason']


def test_outcome_map_marks_true_positives_misses_and_false_alarms():
    from partition_quality_plots import outcome_map
    truth = np.array([[1, 1, 0, 0, 2]])
    predicted = np.array([[1, 0, 1, 0, 1]])
    # TP, FN (missed), FP (outside every object), TN, FN (given to the wrong object)
    assert outcome_map(truth, predicted).tolist() == [[1, 2, 3, 0, 2]]


def test_label_iou_averages_over_objects():
    truth = np.array([[1, 1, 2, 2]])
    predicted = np.array([[1, 0, 2, 2]])
    assert ep.label_iou(truth, predicted, 2) == pytest.approx((.5 + 1) / 2)
    assert ep.label_iou(truth, np.zeros_like(truth), 2) == 0
