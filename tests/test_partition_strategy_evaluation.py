"""Pair strategy outputs before sampling, including weak greedy cases."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import evaluate_partitions as ep


def test_strategy_evaluation_pairs_incomplete_runs_and_reports_weak_cases(tmp_path, monkeypatch):
    images = tmp_path / 'images'
    images.mkdir()
    greedy = 'MSCOCO_sam2_large_254_compact-sam_200'
    split = 'MSCOCO_sam2_large_254_compact-sam_200_split'
    gt = np.zeros((8, 8), dtype=bool)
    gt[:, :2] = True
    for image_id in ['001', '002', '003']:
        Image.new('RGB', (8, 8)).save(images / f'{image_id}.jpg')
        for folder in [greedy, split]:
            if folder == split and image_id == '003':
                continue
            destination = tmp_path / folder
            destination.mkdir(exist_ok=True)
            labels = np.ones((8, 8), dtype=np.uint16)
            if image_id == '002' or folder == split:
                labels[:, 2:] = 2
            np.save(destination / f'{image_id}_compact.npy', labels)
    config = tmp_path / 'demo.yaml'
    config.write_text(f'data:\n  name: MS_COCO\n  image_dir: {images}\n'
                      f'  mask_dir_final: {greedy}\noutput:\n  partitions: {tmp_path}\n')
    monkeypatch.setattr(ep, 'gt_objects', lambda *a: iter([('object', gt, None)]))
    output = tmp_path / 'report'
    args = ep.parser().parse_args(['--config', str(config), '--folders', greedy, split,
                                  '--baseline-folder', greedy, '--no-plots', '--min-images', '1',
                                  '--output-dir', str(output)])
    summaries = ep.run(args)
    assert [s['images'] for s in summaries] == [2, 2]
    pairs = list(csv.DictReader((output / 'strategy_comparison.csv').open()))
    assert {p['image_id'] for p in pairs} == {'001', '002'}
    weak = next(p for p in pairs if p['image_id'] == '001')
    assert weak['weak_greedy'] == 'True'
    assert float(weak['delta_iou']) == pytest.approx(.75)
    assert next(p for p in pairs if p['image_id'] == '002')['weak_greedy'] == 'False'
    assert 'Weak greedy | 1' in (output / 'strategy_comparison.md').read_text()
    # Adding a limit samples the shared pool, not different images from each folder.
    args.limit = 1
    assert [s['images'] for s in ep.run(args)] == [1, 1]


def test_weak_selection_uses_any_bad_object_not_only_image_mean():
    args = ep.parser().parse_args(['--baseline-folder', 'greedy'])
    summaries = [{'config': 'base', 'partitions': 'greedy'}, {'config': 'candidate', 'partitions': 'split'}]
    rows = []
    for label, values in [('base', [.2, 1., 1.]), ('candidate', [.8, 1., 1.])]:
        for value in values:
            rows.append({'config': label, 'image_id': '1', 'best_iou': value, 'cells': 3,
                         'gt_share': 1., 'cause': 'Merged' if value < .5 else ''})
    pairs, _ = ep.strategy_comparison(summaries, rows, args)
    assert pairs[0]['weak_greedy']
    assert pairs[0]['baseline_iou'] > .5
    assert pairs[0]['delta_iou'] == pytest.approx(.2)


def test_strategies_inside_one_folder_are_paired(tmp_path, monkeypatch):
    """Strategies saved as <id>_compact_<tag>.npy in one folder: --partition-types with --baseline-type."""
    images, folder = tmp_path / 'images', tmp_path / 'MSCOCO_sam2_large_254_compact-sam_200'
    images.mkdir()
    folder.mkdir()
    gt = np.zeros((8, 8), dtype=bool)
    gt[:, :2] = True
    for image_id in ['001', '002']:
        Image.new('RGB', (8, 8)).save(images / f'{image_id}.jpg')
        np.save(folder / f'{image_id}_compact_greedy.npy', np.ones((8, 8), dtype=np.uint16))
        split = np.ones((8, 8), dtype=np.uint16)
        split[:, 2:] = 2
        if image_id == '001':
            np.save(folder / f'{image_id}_compact_split.npy', split)
    config = tmp_path / 'demo.yaml'
    config.write_text(f'data:\n  name: MS_COCO\n  image_dir: {images}\n'
                      f'  mask_dir_final: {folder.name}\noutput:\n  partitions: {tmp_path}\n')
    monkeypatch.setattr(ep, 'gt_objects', lambda *a: iter([('object', gt, None)]))
    output = tmp_path / 'report'
    summaries = ep.run(ep.parser().parse_args([
        '--config', str(config), '--partition-types', 'compact_greedy', 'compact_split',
        '--baseline-type', 'compact_greedy', '--no-plots', '--min-images', '1', '--output-dir', str(output)]))
    assert [s['partitions'] for s in summaries] == [folder.name, f'{folder.name}:compact_split']
    assert [s['images'] for s in summaries] == [1, 1]  # Only 001 has both files.
    assert summaries[1]['segmenter'].endswith('-SPLIT')
    pairs = list(csv.DictReader((output / 'strategy_comparison.csv').open()))
    assert len(pairs) == 1 and float(pairs[0]['delta_iou']) == pytest.approx(.75)
