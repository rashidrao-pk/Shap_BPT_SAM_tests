import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import partition_descriptors as pd_  # noqa: E402


def two_colour_image():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:20] = (200, 30, 30)   # top: red
    image[20:] = (30, 30, 200)   # bottom: blue
    return image


def test_aligned_partition_is_homogeneous_and_follows_the_edge():
    image = two_colour_image()
    aligned = np.ones((40, 60), dtype=np.int64)
    aligned[20:] = 2
    crossing = np.ones((40, 60), dtype=np.int64)
    crossing[:, 30:] = 2  # Splits left/right, across the colour edge.
    assert pd_.lab_variance(image, aligned) < 1e-6 < pd_.lab_variance(image, crossing)
    assert pd_.boundary_recall(image, aligned) == 1.0
    assert pd_.boundary_recall(image, crossing) < .2


def test_describe_reads_metadata_and_counts_unassigned_pixels():
    image = two_colour_image()
    partition = np.zeros((40, 60), dtype=np.uint16)
    partition[:20] = 1
    partition[20:, :30] = 2  # The bottom-right quarter stays unassigned (label 0).
    row = pd_.describe(image, partition, {'n_sam_masks': 7, 'mask_target_met': False, 'total_time_sec': 1.5})
    assert row['sam_masks'] == 7 and row['target_met'] is False and row['generation_sec'] == 1.5
    assert row['regions_raw'] == row['regions_after_cap'] == 2
    assert abs(row['unassigned_share'] - .25) < 1e-9
    assert np.isnan(pd_.boundary_recall(np.zeros((10, 10, 3), dtype=np.uint8), partition[:10, :10]))  # No edges.


def test_summary_has_one_row_per_folder():
    rows = [{'dataset': 'X', 'segmenter': 'SAM2-T', 'partitions': 'X_sam2_tiny', 'image_id': str(i), 'regions_after_cap': i,
             'unassigned_share': .1, 'generation_sec': 2., 'lab_variance': 5., 'boundary_recall': .5,
             'target_met': i % 2 == 0} for i in range(4)]
    summary = pd_.summarize(rows)
    assert len(summary) == 1 and summary['images'][0] == 4 and summary['target_met_share'][0] == .5
    assert summary['regions_after_cap_median'][0] == 1.5
