"""view_partitions.py --sam / --partition-folder pick the partition folder."""
import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from view_partitions import load_view_config, resolve_paths  # noqa: E402


def write_config(tmp_path):
    config = tmp_path / 'demo.yaml'
    config.write_text(f'''data:
  name: MSCOCO
  image_dir: {tmp_path}/images
  mask_dir_final: MSCOCO_sam1_vit_b_254_compact-sam_200
  mask_dir_final_sam1: MSCOCO_sam1_vit_b_254_compact-sam_200
  mask_dir_final_sam2: MSCOCO_sam2_large_254_compact-sam_200
output:
  partitions: {tmp_path}/partitions
  evaluations: {tmp_path}/evaluations
''')
    return config


def test_sam_and_folder_choice(tmp_path):
    config = write_config(tmp_path)
    assert load_view_config(str(config))[1]['data']['mask_dir_final'].startswith('MSCOCO_sam1')
    assert load_view_config(str(config), sam='sam2')[1]['data']['mask_dir_final'] == 'MSCOCO_sam2_large_254_compact-sam_200'
    assert load_view_config(str(config), partition_folder='X_62')[1]['data']['mask_dir_final'] == 'X_62'


def test_missing_sam_key_is_reported(tmp_path):
    config = tmp_path / 'plain.yaml'
    config.write_text('data:\n  mask_dir_final: A\n')
    with pytest.raises(ValueError, match='mask_dir_final_sam2'):
        load_view_config(str(config), sam='sam2')


def test_paths_and_preview_folder_follow_the_choice(tmp_path):
    config = write_config(tmp_path)
    (tmp_path / 'images').mkdir()
    (tmp_path / 'images/000000076211.jpg').write_bytes(b'x')
    args = argparse.Namespace(config=str(config), image_id='76211', image=None, partitions=None, output=None,
                              methods=['sam', 'compact'], sam='sam2', partition_folder=None)
    _, partitions, output = resolve_paths(args)
    assert partitions[1] == tmp_path / 'partitions/MSCOCO_sam2_large_254_compact-sam_200/000000076211_compact.npy'
    assert output.parent.name == 'MSCOCO_sam2_large_254_compact-sam_200'


def make_images(tmp_path, ids, folder='MSCOCO_sam2_large_254_compact-sam_200'):
    import numpy as np
    from PIL import Image
    (tmp_path / 'images').mkdir(exist_ok=True)
    (tmp_path / 'partitions' / folder).mkdir(parents=True, exist_ok=True)
    for number in ids:
        stem = f'{number:012d}'
        Image.fromarray(np.full((8, 10, 3), 120, dtype=np.uint8)).save(tmp_path / f'images/{stem}.jpg')
        labels = np.zeros((8, 10), dtype=np.int64)
        labels[:, 5:] = 1
        for method in ('sam', 'compact'):
            np.save(tmp_path / 'partitions' / folder / f'{stem}_{method}.npy', labels)


def test_several_image_ids_render_one_preview_each(tmp_path, monkeypatch, capsys):
    import view_partitions
    config = write_config(tmp_path)
    make_images(tmp_path, [76211, 28285])
    monkeypatch.setattr(sys, 'argv', ['view_partitions.py', '--config', str(config), '--sam', 'sam2',
                                      '--image-id', '76211', '28285', '--methods', 'sam', 'compact'])
    view_partitions.main()
    previews = tmp_path / 'evaluations/previews/MSCOCO_sam2_large_254_compact-sam_200'
    assert sorted(path.name for path in previews.glob('*.png')) == ['000000028285.png', '000000076211.png']


def test_a_missing_image_does_not_stop_the_others(tmp_path, monkeypatch):
    import view_partitions
    config = write_config(tmp_path)
    make_images(tmp_path, [76211])
    monkeypatch.setattr(sys, 'argv', ['view_partitions.py', '--config', str(config), '--sam', 'sam2',
                                      '--image-id', '99999', '76211', '--methods', 'sam', 'compact'])
    with pytest.raises(SystemExit):
        view_partitions.main()
    assert (tmp_path / 'evaluations/previews/MSCOCO_sam2_large_254_compact-sam_200/000000076211.png').is_file()
