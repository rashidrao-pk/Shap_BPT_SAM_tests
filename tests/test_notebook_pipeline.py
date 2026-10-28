import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import yaml

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import sam_notebook_pipeline as pipeline
from run_notebook_pipeline import process
from view_partitions import render, resolve_paths


class NotebookPipelineTest(unittest.TestCase):
    def test_viewer_config_paths_and_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / '000000000139.jpg'
            Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(image)
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({
                'data': {'image_dir': str(root), 'masks_path': str(root),
                         'mask_dir': 'SAM_BPT', 'mask_dir_final': 'large_200'},
                'output': {'dir': str(root), 'folder': 'results'},
            }))
            args = SimpleNamespace(config=str(config), image_id='139', methods=['sam', 'refined'],
                                   image=None, partitions=None, output=None)
            resolved_image, partitions, output = resolve_paths(args)
            self.assertEqual(resolved_image, image)
            self.assertEqual(partitions[0], root / 'SAM_BPT/large_200/000000000139_sam.npy')
            self.assertEqual(output, root / 'results/previews/large_200/000000000139.png')
            for partition in partitions:
                partition.parent.mkdir(parents=True, exist_ok=True)
                np.save(partition, np.zeros((8, 10), dtype=np.int32))
            render(resolved_image, partitions, output)
            self.assertTrue(output.is_file())
            args.image, args.partitions, args.output = image, partitions[:1], root / 'custom.png'
            self.assertEqual(resolve_paths(args), (image, partitions[:1], args.output))
            config.write_text(yaml.safe_dump({
                'data': {'input_format': 'directory', 'image_dir': str(root),
                         'masks_path': str(root), 'mask_dir': 'SAM_BPT', 'mask_dir_final': 'imagenet'},
                'output': {'folder': str(root / 'results')},
            }))
            nested = root / 'nested'
            nested.mkdir()
            image.rename(nested / 'ILSVRC2012_val_00000001.JPEG')
            args.image_id = 'ILSVRC2012_val_00000001'
            args.image = args.partitions = args.output = None
            self.assertEqual(resolve_paths(args)[0], nested / 'ILSVRC2012_val_00000001.JPEG')

    def test_greedy_scores_new_pixels_and_drops_duplicates(self):
        a = np.zeros((6, 6), dtype=bool)
        a[:3] = True
        b = ~a
        annotations = [{'segmentation': a, 'area': 18, 'predicted_iou': 0.8},
                       {'segmentation': a, 'area': 18, 'predicted_iou': 0.9},
                       {'segmentation': b, 'area': 18, 'predicted_iou': 0.7}]
        selected, coverage, history, _ = pipeline.select_masks_by_unpicked_area(annotations, 3)
        self.assertEqual([item['original_mask_id'] for item in history], [1, 2])
        self.assertEqual(len(selected), 2)
        self.assertTrue(coverage.all())

    def test_compact_booster_fill_and_inverse(self):
        big = np.ones((7, 7), dtype=bool)
        small = np.zeros((7, 7), dtype=bool)
        small[2:5, 2:5] = True
        masks = [{'segmentation': big, 'area': 49}, {'segmentation': small, 'area': 9}]
        compact = pipeline.create_compact_mask(masks)
        self.assertEqual(compact[3, 3], 2)
        self.assertEqual(compact[0, 0], 1)
        inverse = pipeline.compact_mask_to_masks(compact)
        self.assertEqual(sum(mask['area'] for mask in inverse), 49)
        boosted = pipeline.refiner(compact, masks)
        np.testing.assert_array_equal(compact, boosted)
        labels = np.array([[1, 0, 0, 0, 2]])
        filled = pipeline.partition_unassigned_by_nearest_cluster(labels, max_distance=1)
        np.testing.assert_array_equal(filled, [[1, 1, 0, 2, 2]])
        tied = pipeline.partition_unassigned_by_nearest_cluster(labels, keep_ties_unassigned=True)
        self.assertEqual(tied[0, 2], 0)

    def test_end_to_end_empty_masks_and_preview_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'sample.png'
            Image.fromarray(np.zeros((16, 20, 3), dtype=np.uint8)).save(image)
            metadata = process(image, np.empty((0, 16, 20), dtype=bool), [], root / 'empty')
            self.assertEqual(metadata['n_sam_masks'], 0)
            self.assertTrue((root / 'empty/sample_partitions.png').is_file())
            self.assertEqual(metadata['statistics']['sam']['coverage_percent'], 0)
            invalid = root / 'invalid.npy'
            np.save(invalid, np.zeros((2, 2), dtype=int))
            with self.assertRaisesRegex(ValueError, 'expected integer labels'):
                render(image, [invalid], root / 'invalid.png')

    def test_gif_and_step_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'sample.png'
            Image.fromarray(np.zeros((16, 20, 3), dtype=np.uint8)).save(image)
            masks = np.zeros((2, 16, 20), dtype=bool)
            masks[0, :8] = True
            masks[1, 8:] = True
            with contextlib.redirect_stdout(io.StringIO()):
                metadata = process(image, masks, [0.9, 0.8], root / 'output',
                                   create_gif=True, save_step_images=True)
            self.assertTrue(Path(metadata['gif']).is_file())
            self.assertTrue(list((root / 'output/sample_steps').glob('*.png')))
            self.assertEqual(metadata['statistics']['boosted']['coverage_percent'], 100)
            self.assertTrue((root / 'output/sample_summary.csv').is_file())


if __name__ == '__main__':
    unittest.main()
