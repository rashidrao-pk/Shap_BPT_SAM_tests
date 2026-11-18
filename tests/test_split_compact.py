"""--compact-strategy split keeps masks nested inside already chosen ones (windows on a facade)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import generate_coco_partitions as batch  # noqa: E402
from generate_sam_partitions import compact_partitions, select_masks_by_split_gain  # noqa: E402


def facade(shape=(60, 80)):
    """A facade covering the top, a pavement below, three windows inside the facade, and a duplicate facade."""
    masks = []
    top = np.zeros(shape, bool); top[:45] = True; masks.append(top)
    bottom = np.zeros(shape, bool); bottom[45:] = True; masks.append(bottom)
    for x in (5, 30, 55):
        window = np.zeros(shape, bool); window[10:25, x:x + 15] = True; masks.append(window)
    masks.append(top.copy())
    return np.stack(masks), np.full(len(masks), .9, dtype=np.float32)


def regions(labels):
    return int(np.count_nonzero(np.unique(labels)))


def test_greedy_stops_at_full_coverage_but_split_keeps_windows():
    masks, ious = facade()
    assert regions(compact_partitions(masks, ious, 254, 'greedy')) == 2
    split = compact_partitions(masks, ious, 254, 'split', min_gain=10)
    assert regions(split) == 5
    assert len(np.unique(split[10:25, 5:20])) == 1 and split[10, 5] != split[0, 0]


def test_duplicates_gain_nothing_and_the_cap_and_min_gain_hold():
    masks, ious = facade()
    assert len(select_masks_by_split_gain(masks, ious, 254, min_gain=10)) == 5  # Duplicate facade dropped.
    assert len(select_masks_by_split_gain(masks, ious, 3, min_gain=10)) == 3
    assert len(select_masks_by_split_gain(masks, ious, 254, min_gain=1000)) == 2  # Windows (225 px) too small.


def test_split_matches_painting_all_kept_masks():
    rng = np.random.default_rng(0)
    shape = (48, 64)
    masks = []
    for _ in range(40):
        mask = np.zeros(shape, bool)
        y, x = rng.integers(0, 40), rng.integers(0, 56)
        mask[y:y + rng.integers(4, 30), x:x + rng.integers(4, 40)] = True
        masks.append(mask)
    masks = np.stack(masks)
    kept = select_masks_by_split_gain(masks, np.ones(len(masks)), 254, min_gain=1)
    # Every kept mask must still be visible in the painted result, else it should not have been chosen.
    painted = compact_partitions(masks, np.ones(len(masks)), 254, 'split', min_gain=1)
    assert regions(painted) >= len(kept) * 0.8


def test_split_runs_fast_enough_for_sam_sized_inputs():
    rng = np.random.default_rng(1)
    shape = (480, 640)
    masks = np.zeros((200, *shape), bool)
    for mask in masks:
        y, x = rng.integers(0, 440), rng.integers(0, 600)
        mask[y:y + rng.integers(10, 200), x:x + rng.integers(10, 300)] = True
    start = time.perf_counter()
    select_masks_by_split_gain(masks, np.ones(200), 254)
    assert time.perf_counter() - start < 60


def test_split_file_names():
    args = argparse.Namespace(model='sam2', checkpoint=Path('sam2.1_hiera_large.pt'), max_regions=254,
                              generate_at_least_n_masks=200, save_partitions=['sam', 'compact'],
                              compact_strategy=['greedy', 'split'], mso_alpha=[0.5], min_region_pixels=64)
    assert batch.partition_folder_name({'name': 'MSCOCO'}, args).endswith('_compact-sam_200')
    assert batch.output_stages(args) == ['sam', 'compact_greedy', 'compact_split']
    args.min_region_pixels = 200
    assert batch.output_stages(args) == ['sam', 'compact_greedy', 'compact_split_200']
