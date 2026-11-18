"""compare_partition_folders.py: compact strategies saved in one folder, or several folders, side by side."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import compare_partition_folders as compare  # noqa: E402
import evaluate_partitions  # noqa: E402

FOLDER = 'MSCOCO_sam2_large_254_compact-sam_200'
SAM1 = 'MSCOCO_sam1_vit_b_254_compact-sam_200'


def test_labels_for_strategies_and_older_suffixed_folders():
    assert compare.type_label('compact_greedy') == 'Greedy'
    assert compare.type_label('compact') == 'Greedy'  # Legacy file name
    assert compare.type_label('compact_mso_0.5') == 'MSO α=0.5'
    assert compare.type_label('compact_split') == 'Split'
    assert compare.type_label('compact_split_200') == 'Split (gain ≥200 px)'
    # Folders written while strategies were folder suffixes:
    assert compare.folder_label(f'{FOLDER}_greedy') == 'Greedy'
    assert compare.folder_label(f'{FOLDER}_mso_0.25') == 'MSO α=0.25'
    assert compare.folder_label(f'{FOLDER}_split_200') == 'Split (gain ≥200 px)'
    assert compare.folder_label('custom') == 'custom'


def make_dataset(tmp_path):
    images, partitions = tmp_path / 'images', tmp_path / 'partitions'
    images.mkdir()
    shape = (20, 30)
    Image.fromarray(np.full((*shape, 3), 120, dtype=np.uint8)).save(images / '000000076211.jpg')
    sam = (np.arange(np.prod(shape)).reshape(shape) % 7 + 1).astype(np.uint16)
    left = np.zeros(shape, dtype=np.uint16)
    left[:, :15] = 1                          # matches the GT object exactly
    coarse = np.ones(shape, dtype=np.uint16)  # one region for the whole image
    coarse[0, 0] = 0
    folder = partitions / FOLDER
    folder.mkdir(parents=True)
    np.save(folder / '000000076211_sam.npy', sam)
    np.save(folder / '000000076211_compact_greedy.npy', coarse)
    np.save(folder / '000000076211_compact_mso_0.5.npy', left)
    np.save(folder / '000000099999_compact_greedy.npy', coarse)  # no MSO file for this image
    (partitions / SAM1).mkdir()
    np.save(partitions / SAM1 / '000000076211_compact.npy', left)  # legacy name, read as greedy
    config = tmp_path / 'demo.yaml'
    config.write_text(f'''data:
  name: MSCOCO
  image_dir: {images}
  mask_dir_final: {FOLDER}
output:
  partitions: {partitions}
  evaluations: {tmp_path}/evaluations
''')
    return config, shape


def test_ids_must_exist_for_every_entry(tmp_path):
    make_dataset(tmp_path)
    folder = tmp_path / 'partitions' / FOLDER
    entries = [(folder, 'compact_greedy'), (folder, 'compact_mso_0.5')]
    assert compare.select_ids(entries) == ['000000076211']
    assert compare.select_ids(entries, None, ['76211', '99999']) == ['000000076211']
    assert compare.select_ids([folder], 'compact_greedy') == ['000000076211', '000000099999']


def test_strategies_in_one_folder(tmp_path, monkeypatch):
    config, shape = make_dataset(tmp_path)
    gt = np.zeros(shape, dtype=bool)
    gt[:, :15] = True
    monkeypatch.setattr(evaluate_partitions, 'gt_objects', lambda *args, **kwargs: iter([('person', gt, None)]))
    summary = compare.main(['--config', str(config), '--partition-types', 'compact_greedy', 'compact_mso_0.5',
                            '--image-id', '76211'])
    by_label = {row['label']: row for row in summary}
    assert by_label['MSO α=0.5']['mean_iou'] == 1.0
    assert by_label['Greedy']['mean_iou'] < 0.6
    assert by_label['MSO α=0.5']['best_on_images'] == 1 and by_label['Greedy']['best_on_images'] == 0
    assert by_label['Greedy']['mean_regions'] == 1
    output = tmp_path / 'evaluations/partition_compare/compare'
    assert (output / '000000076211.png').is_file()
    rows = list(csv.DictReader((output / 'per_image.csv').open()))
    assert [row['label'] for row in rows] == ['Greedy', 'MSO α=0.5']
    assert [row['partition_type'] for row in rows] == ['compact_greedy', 'compact_mso_0.5']
    assert rows[0]['sam_regions'] == '7' and rows[1]['regions'] == '1'
    assert float(rows[0]['unassigned_pct']) > 0


def test_folders_with_the_configured_strategy(tmp_path, monkeypatch):
    config, shape = make_dataset(tmp_path)
    gt = np.zeros(shape, dtype=bool)
    gt[:, :15] = True
    monkeypatch.setattr(evaluate_partitions, 'gt_objects', lambda *args, **kwargs: iter([('person', gt, None)]))
    summary = compare.main(['--config', str(config), '--folders', FOLDER, SAM1, '--labels', 'SAM2-L', 'SAM1-VB',
                            '--no-figures'])
    by_label = {row['label']: row for row in summary}
    assert by_label['SAM1-VB']['mean_iou'] == 1.0 and by_label['SAM2-L']['mean_iou'] < .6
