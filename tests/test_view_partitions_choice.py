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
