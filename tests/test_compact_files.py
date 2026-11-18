"""One partition folder, one file per compact strategy: <id>_compact_greedy.npy, <id>_compact_split.npy, ..."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import generate_coco_partitions as batch  # noqa: E402
import partition_files as pf  # noqa: E402
from utils_sam_2 import load_refine_partitions  # noqa: E402


def facade_masks(shape=(40, 60)):
    """A facade, a pavement and a window inside the facade: greedy keeps 2 regions, split keeps 3."""
    top = np.zeros(shape, bool); top[:30] = True
    bottom = np.zeros(shape, bool); bottom[30:] = True
    window = np.zeros(shape, bool); window[5:20, 10:30] = True
    return np.stack([top, bottom, window])


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    images = tmp_path / 'images'
    images.mkdir()
    for name in ['a', 'b']:
        Image.fromarray(np.full((40, 60, 3), 100, np.uint8)).save(images / f'{name}.png')
    config = tmp_path / 'demo.yaml'
    config.write_text(f'data:\n  name: Demo\n  input_format: directory\n  image_dir: {images}\n'
                      f'output:\n  partitions: {tmp_path / "partitions"}\n')
    calls = []

    def infer(image, args, loaded=None, return_history=False, return_annotations=False):
        calls.append(1)
        masks = facade_masks(image.shape[:2])
        annotations = [{'segmentation': m, 'predicted_iou': .9} for m in masks]
        result = masks, .1, 'cpu', 'fake.pt'
        return (*result, [], annotations) if return_annotations else (*result, []) if return_history else result

    monkeypatch.setattr(batch.sam, 'build_generator', lambda args: (object(), 'cpu', 'fake.pt'))
    monkeypatch.setattr(batch.sam, 'infer_masks', infer)
    return config, tmp_path / 'partitions' / 'Demo_sam2_large_254_compact-sam_singlepass', calls


def run(config, *extra):
    args = batch.create_parser().parse_args(['--config', str(config), '--checkpoint', 'sam2.1_hiera_large.pt',
                                             '--device', 'cpu', '--save-partitions', 'sam', 'compact', *extra])
    batch._run(args, sys.stderr)


def regions(path):
    return int(np.count_nonzero(np.unique(np.load(path))))


def test_one_run_writes_every_strategy_into_one_folder(dataset):
    config, folder, calls = dataset
    run(config, '--compact-strategy', 'greedy', 'split', 'mso', '--mso-alpha', '0.5')
    assert len(calls) == 2  # SAM once per image for all strategies
    names = sorted(path.name for path in folder.iterdir() if path.name.startswith('a_'))
    assert names == ['a_compact_greedy.npy', 'a_compact_mso_0.5.npy', 'a_compact_split.npy', 'a_results.json',
                     'a_sam.npy']
    assert regions(folder / 'a_compact_greedy.npy') == 2 and regions(folder / 'a_compact_split.npy') == 3
    metadata = json.loads((folder / 'a_results.json').read_text())
    assert set(metadata['compact_files']) == {'greedy', 'split', 'mso_0.5'}
    assert metadata['compact_files']['split']['regions'] == 3
    assert 'compact_strategy' not in metadata['generation_settings']


def test_a_later_strategy_is_added_without_touching_existing_files(dataset):
    config, folder, calls = dataset
    run(config)  # greedy only
    before = (folder / 'a_compact_greedy.npy').stat().st_mtime_ns
    run(config)  # nothing to do
    assert len(calls) == 2
    run(config, '--compact-strategy', 'greedy', 'split')
    assert len(calls) == 4  # SAM again, only for the missing split files
    assert (folder / 'a_compact_greedy.npy').stat().st_mtime_ns == before
    metadata = json.loads((folder / 'a_results.json').read_text())
    assert set(metadata['saved_partitions']) == {'sam', 'compact_greedy', 'compact_split'}
    assert set(metadata['compact_files']) == {'greedy', 'split'} and 'compact_split' in metadata['added_stages']


def test_legacy_compact_files_count_as_greedy(dataset):
    config, folder, calls = dataset
    run(config)
    for name in ['a', 'b']:
        (folder / f'{name}_compact_greedy.npy').rename(folder / f'{name}_compact.npy')
    run(config)
    assert len(calls) == 2  # Legacy greedy files satisfy the greedy request
    image = np.zeros((40, 60, 3), np.uint8)
    assert load_refine_partitions(image, str(folder), 'a', 'compact_greedy')[0].shape == (40, 60)
    assert set(pf.stage_files(folder, 'compact_greedy')) == {'a', 'b'}


def test_names_and_config_selection():
    assert pf.compact_tags(['greedy', 'mso', 'split'], [0.3, 0.5], 200) == ['greedy', 'mso_0.3', 'mso_0.5', 'split_200']
    assert pf.stage('split') == 'compact_split' and pf.tag_of('compact_mso_0.5') == 'mso_0.5'
    assert pf.configured_stage({}) == 'compact_greedy'
    assert pf.configured_stage({'compact_strategy': 'split'}) == 'compact_split'
    assert pf.run_label('X_compact-sam_200', 'greedy') == 'X_compact-sam_200'
    assert pf.run_label('X_compact-sam_200', 'split') == 'X_compact-sam_200_split'
    assert pf.tag_label('compact_split') == 'SPLIT' and pf.tag_label('compact_greedy') == ''


def test_runner_partition_choice():
    import argparse
    parser = pf.add_partition_arguments(argparse.ArgumentParser())
    config = {'data': {'mask_dir_final': 'X_sam1_vit_b_254_compact-sam_200',
                       'mask_dir_final_sam2': 'X_sam2_large_254_compact-sam_200'}}
    args = parser.parse_args(['--sam', 'sam2', '--compact-strategy', 'split'])
    line = pf.apply_partition_choice(config, args.partition_sam, args.partition_folder, args.partition_strategy)
    assert config['data']['mask_dir_final'] == 'X_sam2_large_254_compact-sam_200'
    assert config['data']['compact_strategy'] == 'split' and 'compact_split' in line
    with pytest.raises(ValueError, match='mask_dir_final_medsam'):
        pf.apply_partition_choice(config, 'medsam')
    with pytest.raises(SystemExit):
        parser.parse_args(['--sam', 'sam2', '--partition-folder', 'Y'])  # one or the other
    with pytest.raises(ValueError):
        pf.apply_partition_choice(config, compact_strategy='random')
