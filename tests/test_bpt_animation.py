"""Animation cuts must be nested cuts of the actual explanation trees."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import animate_bpt_hierarchies as animation


def test_cut_sequence_is_nested_and_covers_every_pixel():
    import shap_bpt
    image = np.random.default_rng(7).integers(0, 256, (9, 12, 3), dtype=np.uint8)
    guidance = np.indices((9, 12))[1].astype(np.int32) // 4
    for kwargs in [{}, {'prebuilt_partitions': guidance}]:
        tree = shap_bpt.build_bpt_from_image(image, **kwargs)
        previous = None
        for count in [108, 20, 8, 2, 1]:
            labels, anchors = animation.tree_cut(tree, count)
            assert labels.shape == image.shape[:2]
            assert len(np.unique(labels)) == len(anchors) == count
            if previous is not None:
                for label in np.unique(previous):
                    assert len(np.unique(labels[previous == label])) == 1
            previous = labels
        with pytest.raises(ValueError):
            animation.tree_cut(tree, 109)


def test_schedule_descends_to_root_without_duplicate_counts():
    counts = animation.region_schedule(30, 256, 48)
    assert counts[0] == 30 and counts[-1] == 1
    assert all(a > b for a, b in zip(counts, counts[1:]))


def test_gif_contains_distinct_frames_and_static_preview(tmp_path):
    import shap_bpt
    image = np.random.default_rng(2).integers(0, 256, (12, 16, 3), dtype=np.uint8)
    labels = np.indices((12, 16))[1].astype(np.int32) // 4
    trees = [shap_bpt.build_bpt_from_image(image),
             shap_bpt.build_bpt_from_image(image, prebuilt_partitions=labels)]
    output = tmp_path / 'example.gif'
    counts = animation.render(image, labels, trees, output, 'Test input',
                              SimpleNamespace(start_regions=16, frames=5, fps=5, dpi=45))
    with Image.open(output) as gif:
        assert gif.n_frames == len(counts)
        assert gif.info['loop'] == 0
        gif.seek(0)
        first = np.asarray(gif.convert('RGB'))
        gif.seek(gif.n_frames - 1)
        assert not np.array_equal(first, np.asarray(gif.convert('RGB')))
    assert output.with_suffix('.png').is_file()
