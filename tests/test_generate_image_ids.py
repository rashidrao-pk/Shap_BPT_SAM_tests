"""--image-id for the partition generators (shared by every generate_*_partitions.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import generate_coco_partitions as gen  # noqa: E402

COCO = [{'id': 139, 'file_name': '000000000139.jpg'}, {'id': 76211, 'file_name': '000000076211.jpg'},
        {'id': 285, 'file_name': '000000000285.jpg'}]
DIRECTORY = [{'id': 'camourflage_00012', 'file_name': 'Imgs/camourflage_00012.jpg'},
             {'id': 'benign (12)', 'file_name': 'benign/benign (12).png'}]


def test_numeric_ids_match_padded_and_unpadded():
    assert gen.select_image_ids(COCO, ['76211']) == [COCO[1]]
    assert gen.select_image_ids(COCO, ['000000000139', '285']) == [COCO[0], COCO[2]]  # Dataset order kept.


def test_directory_ids_match_by_stem():
    assert gen.select_image_ids(DIRECTORY, ['benign (12)']) == [DIRECTORY[1]]
    assert gen.select_image_ids(DIRECTORY, ['camourflage_00012']) == [DIRECTORY[0]]


def test_unknown_ids_are_reported():
    with pytest.raises(ValueError, match='99999'):
        gen.select_image_ids(COCO, ['139', '99999'])


def test_flag_parses_and_does_not_change_resume_settings():
    parser = gen.create_parser()
    single = parser.parse_args(['--image-id', '76211', '139'])
    full = parser.parse_args([])
    assert single.image_ids == ['76211', '139']
    assert gen.generation_settings(single) == gen.generation_settings(full)
    assert gen.partition_folder_name({'name': 'MSCOCO'}, single) == gen.partition_folder_name({'name': 'MSCOCO'}, full)


def test_resume_stamps_ignore_the_compact_strategy():
    """Strategies are separate files in one folder, so any stamp (old, greedy, split, mso) matches."""
    from generate_coco_partitions import same_settings
    old = {'model': 'sam2', 'max_regions': 254}
    greedy = {**old, 'compact_strategy': 'greedy', 'mso_alpha': 0.5}
    assert same_settings(old, greedy)
    assert same_settings(old, {**old, 'compact_strategy': ['mso', 'split'], 'mso_alpha': [0.3]})
    assert not same_settings(old, {**greedy, 'max_regions': 62})
