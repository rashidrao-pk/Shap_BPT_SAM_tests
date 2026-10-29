"""Dataset GT decoding, class matching, nearest-neighbor alignment, and preview."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from ground_truth import load_ground_truth, save_ground_truth, _mask_index, _categories, _classifier_synsets
from view_partitions import main as preview_main


class GroundTruthTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        _mask_index.cache_clear()
        _categories.cache_clear()
        _classifier_synsets.cache_clear()

    def config(self, format_name, **settings):
        return {'data': {'name': 'test', 'ground_truth': {
            'format': format_name, 'root': str(self.root / 'gt'), **settings}}}

    def write(self, relative, array):
        path = self.root / 'gt' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)
        return path

    def test_camo_split_and_nearest_resize(self):
        self.write('Train/camourflage_00001.png', np.array([[0, 255], [0, 0]], dtype=np.uint8))
        self.write('Test/camourflage_00001.png', np.full((2, 2), 255, dtype=np.uint8))
        gt = load_ground_truth(self.config('binary', split='Train'), 'camourflage_00001.jpg', (4, 4, 3))
        self.assertEqual(gt['foreground_pixels'], 4)
        self.assertTrue(gt['resized'])
        self.assertEqual(gt['count_label'], 'components')
        self.assertEqual(gt['scope'], 'dataset_foreground')
        self.assertIsNone(gt['instance_count'])
        self.assertTrue(gt['mask'][:2, 2:].all())

    def test_celeba_group_padding_and_part_selection(self):
        self.write('1/02001_hair.png', np.array([[0, 255, 0], [0, 0, 0]], dtype=np.uint8))
        self.write('1/02001_skin.png', np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8))
        config = self.config('celebamask_hq')
        hair = load_ground_truth(config, '2001.jpg', (4, 6, 3), gt_class='hair')
        self.assertEqual(hair['foreground_pixels'], 4)
        self.assertEqual(hair['part_names'], ['hair'])
        self.assertEqual(hair['display_count'], 1)
        all_parts = load_ground_truth(config, '2001.jpg', (4, 6, 3))
        self.assertEqual(all_parts['foreground_pixels'], 8)
        with self.assertRaisesRegex(ValueError, 'Unknown CelebAMask'):
            load_ground_truth(config, '2001.jpg', (4, 6, 3), gt_class='chair')

    def imagenets_fixture(self):
        labels = np.array([[0, 300, 1000], [1, 300, 0]], dtype=np.uint16)
        encoded = np.zeros((2, 3, 3), dtype=np.uint8)
        encoded[:, :, 0] = labels % 256
        encoded[:, :, 1] = labels // 256
        self.write('ImageNetS919/validation-segmentation/n00000300/ILSVRC2012_val_00000001.png', encoded)
        return self.config('imagenets', variant=919)

    def test_imagenets_rgb_decode_ignore_and_numeric_label(self):
        config = self.imagenets_fixture()
        gt = load_ground_truth(config, 'ILSVRC2012_val_00000001.JPEG', (2, 3, 3))
        self.assertEqual(gt['foreground_pixels'], 3)
        self.assertEqual(gt['ignored_pixels'], 1)
        self.assertEqual(gt['label_map'][0, 1], 300)
        selected = load_ground_truth(config, 'ILSVRC2012_val_00000001.JPEG', (2, 3, 3), gt_class='300')
        self.assertEqual(selected['foreground_pixels'], 2)
        self.assertFalse(selected['mask'][1, 0])

    def test_imagenets_classifier_index_is_not_segmentation_id(self):
        config = self.imagenets_fixture()
        categories = self.root / 'categories.txt'
        categories.write_text('\n'.join(f'n{index:08d}' for index in range(1, 920)))
        mapping = self.root / 'class_index.json'
        mapping.write_text(json.dumps({'0': ['n00000300', 'target']}))
        config['data']['ground_truth'].update(categories_file=str(categories), class_index_file=str(mapping))
        gt = load_ground_truth(config, 'ILSVRC2012_val_00000001.JPEG', (2, 3, 3), predicted_index=0)
        self.assertEqual(gt['category_ids'], [300])
        self.assertEqual(gt['foreground_pixels'], 2)
        self.assertEqual(gt['scope'], 'selected_class')

    def test_missing_annotations_and_mapping_are_explicit(self):
        config = self.imagenets_fixture()
        gt = load_ground_truth(config, 'ILSVRC2012_val_00000001.JPEG', (2, 3, 3),
                               predicted_index=0, label_space='custom')
        self.assertEqual(gt['scope'], 'mapping_unavailable')
        self.assertFalse(gt['mask'].any())
        missing = load_ground_truth(config, 'ILSVRC2012_val_99999999.JPEG', (2, 3, 3))
        self.assertEqual(missing['status'], 'annotation not available')
        self.assertFalse(missing['mask'].any())

    def test_ambiguous_masks_are_rejected(self):
        self.write('Train/a/x.png', np.zeros((2, 2), dtype=np.uint8))
        self.write('Train/b/x.png', np.zeros((2, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'Ambiguous GT'):
            load_ground_truth(self.config('binary'), 'x.jpg', (2, 2, 3))

    def test_celeba_gt_preview_cli_and_saved_metadata(self):
        self.write('0/00000_hair.png', np.array([[0, 255], [0, 0]], dtype=np.uint8))
        images = self.root / 'images'
        images.mkdir()
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / '0.jpg')
        partitions = self.root / 'partitions/face'
        partitions.mkdir(parents=True)
        np.save(partitions / '0_compact.npy', np.zeros((4, 4), dtype=np.uint16))
        config = self.config('celebamask_hq')
        config['data'].update(name='CelebAHQ', input_format='directory', image_dir=str(images), mask_dir_final='face')
        config['output'] = {'partitions': str(partitions.parent), 'explanations': str(self.root / 'x'),
                            'evaluations': str(self.root / 'e')}
        path = self.root / 'config.yaml'
        path.write_text(yaml.safe_dump(config))
        args = ['view_partitions.py', '--config', str(path), '--image-id', '0', '--methods', 'compact',
                '--show_gt', '--gt-class', 'hair']
        with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
            preview_main()
        output = self.root / 'e/previews/face'
        self.assertTrue((output / '0.png').is_file())
        self.assertTrue((output / '0_gt_labels.npz').is_file())
        metadata = json.loads((output / '0_gt.json').read_text())
        self.assertNotIn('mask', metadata)
        self.assertNotIn('label_map', metadata)
        self.assertEqual(metadata['class_name'], 'hair')
        self.assertEqual(metadata['foreground_pixels'], 4)


if __name__ == '__main__':
    unittest.main()
