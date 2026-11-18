"""compare_partition_folders.py: side-by-side comparison of folders such as greedy vs MSO compact partitions."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import compare_partition_folders as compare  # noqa: E402
import evaluate_partitions  # noqa: E402

GREEDY = 'MSCOCO_sam2_large_254_compact-sam_200_greedy'
MSO = 'MSCOCO_sam2_large_254_compact-sam_200_mso_0.5'


def test_labels_come_from_the_folder_suffix():
    assert compare.folder_label(GREEDY) == 'Greedy'
    assert compare.folder_label(MSO) == 'MSO α=0.5'
    assert compare.folder_label('MSCOCO_mso_0.25') == 'MSO α=0.25'
    assert compare.folder_label('custom') == 'custom'


def make_dataset(tmp_path):
    images, partitions = tmp_path / 'images', tmp_path / 'partitions'
    images.mkdir()
    shape = (20, 30)
    Image.fromarray(np.full((*shape, 3), 120, dtype=np.uint8)).save(images / '000000076211.jpg')
    sam = np.arange(np.prod(shape)).reshape(shape) % 7 + 1
    left = np.zeros(shape, dtype=np.uint16)
    left[:, :15] = 1                       # matches the GT object exactly
    coarse = np.ones(shape, dtype=np.uint16)  # one region for the whole image
    coarse[0, 0] = 0
    for name, compact in [(GREEDY, coarse), (MSO, left)]:
        (partitions / name).mkdir(parents=True)
        np.save(partitions / name / '000000076211_compact.npy', compact)
        np.save(partitions / name / '000000076211_sam.npy', sam.astype(np.uint16))
    np.save(partitions / GREEDY / '000000099999_compact.npy', coarse)  # only in one folder
    config = tmp_path / 'demo.yaml'
    config.write_text(f'''data:
  name: MSCOCO
  image_dir: {images}
  mask_dir_final: {GREEDY}
output:
  partitions: {partitions}
  evaluations: {tmp_path}/evaluations
''')
    return config, shape


def test_ids_must_exist_in_every_folder(tmp_path):
    config, _ = make_dataset(tmp_path)
    folders = [tmp_path / 'partitions' / GREEDY, tmp_path / 'partitions' / MSO]
    assert compare.select_ids(folders, 'compact') == ['000000076211']
    assert compare.select_ids(folders, 'compact', ['76211', '99999']) == ['000000076211']


def test_comparison_scores_tables_and_figure(tmp_path, monkeypatch):
    config, shape = make_dataset(tmp_path)
    gt = np.zeros(shape, dtype=bool)
    gt[:, :15] = True
    monkeypatch.setattr(evaluate_partitions, 'gt_objects', lambda *args, **kwargs: iter([('person', gt, None)]))
    summary = compare.main(['--config', str(config), '--folders', GREEDY, MSO, '--image-id', '76211'])
    by_label = {row['label']: row for row in summary}
    assert by_label['MSO α=0.5']['mean_iou'] == 1.0
    assert by_label['Greedy']['mean_iou'] < 0.6
    assert by_label['MSO α=0.5']['best_on_images'] == 1 and by_label['Greedy']['best_on_images'] == 0
    assert by_label['Greedy']['mean_regions'] == 1
    output = tmp_path / 'evaluations/partition_compare/compare'
    assert (output / '000000076211.png').is_file()
    rows = list(csv.DictReader((output / 'per_image.csv').open()))
    assert [row['label'] for row in rows] == ['Greedy', 'MSO α=0.5']
    assert rows[0]['sam_regions'] == '7' and rows[1]['regions'] == '1'
    assert float(rows[0]['unassigned_pct']) > 0
