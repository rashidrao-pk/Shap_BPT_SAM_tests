"""Animation cuts must be nested cuts of the actual explanation trees."""
import sys
import json
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


def test_cli_reads_configured_split_stage_and_writes_metadata_and_gallery(tmp_path):
    images = tmp_path / 'images'
    folder = tmp_path / 'partitions' / 'demo_sam2'
    images.mkdir()
    folder.mkdir(parents=True)
    Image.fromarray(np.random.default_rng(4).integers(0, 256, (12, 16, 3), dtype=np.uint8)).save(images / '000001.jpg')
    np.save(folder / '000001_compact_split.npy', np.indices((12, 16))[1].astype(np.int32) // 4)
    config = tmp_path / 'demo.yaml'
    config.write_text(f'data:\n  name: MS_COCO\n  image_dir: {images}\n  mask_dir_final: demo_sam2\n'
                      f'  compact_strategy: split\noutput:\n  partitions: {tmp_path / "partitions"}\n'
                      f'  evaluations: {tmp_path / "evaluations"}\n')
    outputs = animation.main(['--config', str(config), '--image-id', '1', '--frames', '3', '--dpi', '30',
                              '--start-regions', '8', '--max-side', '8'])
    assert len(outputs) == 1
    metadata = json.loads(Path(outputs[0]).with_suffix('.json').read_text())
    assert metadata['original_shape'] == [12, 16]
    assert metadata['animated_shape'] == [6, 8]
    assert metadata['partition_file'].endswith('_compact_split.npy')
    assert metadata['region_counts'][-1] == 1
    assert metadata['guide_settings']['guide_alpha'] == 2.
    assert (tmp_path / 'evaluations/bpt_animations/index.html').is_file()
