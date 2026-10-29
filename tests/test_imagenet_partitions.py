"""ImageNet discovery, naming, logging, and resume without GPU inference."""
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
import generate_coco_partitions as batch
import generate_imagenet_partitions as imagenet


class ImageNetExportTest(unittest.TestCase):
    def test_imagenet_entrypoint_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images'
            (images / 'nested').mkdir(parents=True)
            rgb = np.zeros((8, 10, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(images / 'ILSVRC2012_val_00000002.JPEG')
            Image.fromarray(rgb).save(images / 'nested/ILSVRC2012_val_00000001.JPEG')
            (images / 'README.txt').write_text('Ignore non-image files')
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'data': {
                'name': 'ImageNet', 'input_format': 'directory', 'image_dir': str(images),
                'masks_path': str(root), 'mask_dir': 'partitions',
            }}))
            with patch.object(sys, 'argv', ['generate_imagenet_partitions.py']):
                self.assertEqual(imagenet.parse_args().config, 'ImageNet_epito')
            args = batch.create_parser().parse_args(['--config', str(config), '--limit', '1'])
            masks = np.ones((1, 8, 10), dtype=bool)
            with patch.object(batch.sam, 'build_generator', return_value=('fake', 'cpu', 'fake.pt')) as build:
                with patch.object(batch.sam, 'infer_masks', return_value=(masks, 0.2, 'cpu', 'fake.pt')):
                    imagenet.run(args)
                    output = root / 'partitions/ImageNet_sam2_tiny_62_refined_singlepass'
                    prefix = output / 'ILSVRC2012_val_00000002'
                    # Ordering follows relative paths, as documented.
                    self.assertTrue(prefix.with_name(prefix.name + '_refined.npy').is_file())
                    metadata = json.loads(prefix.with_name(prefix.name + '_results.json').read_text())
                    self.assertEqual(metadata['source_image'], str(images / 'ILSVRC2012_val_00000002.JPEG'))
                    self.assertEqual(np.load(prefix.with_name(prefix.name + '_refined.npy')).shape, (8, 10))
                    imagenet.run(args)
                    self.assertEqual(build.call_count, 1)
                    args.limit = None
                    imagenet.run(args)
                    self.assertEqual(build.call_count, 2)
                    self.assertTrue((output / 'ILSVRC2012_val_00000001_refined.npy').is_file())
                    log = output.with_suffix('.log').read_text()
                    self.assertIn('ImageNet_sam2_tiny_62_refined_singlepass', log)
                    self.assertIn('1 masks, 1 regions', log)
                    args.dry_run = True
                    imagenet.run(args)
                    self.assertEqual(build.call_count, 2)

    def test_gt_selection_and_resume_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks = root / 'ImageNetS50/validation-segmentation/n00000001'
            masks.mkdir(parents=True)
            Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(masks / 'annotated.png')
            Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(masks / 'missing_source.png')
            data = {'name': 'ImageNet', 'ground_truth': {
                'format': 'imagenets', 'root': str(root), 'variant': 50}}
            images = [{'file_name': 'first.JPEG'}, {'file_name': 'nested/annotated.JPEG'}]
            self.assertEqual(batch.select_gt_images(data, images), [images[1]])
            args = batch.create_parser().parse_args([])
            before = batch.generation_settings(args)
            args.gt_only = True
            self.assertEqual(batch.generation_settings(args), before)

    def test_duplicate_stems_fail_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'a').mkdir()
            (root / 'b').mkdir()
            for directory in ['a', 'b']:
                (root / directory / 'same.JPEG').touch()
            with self.assertRaisesRegex(ValueError, 'Duplicate image stem'):
                batch.collect_images({'input_format': 'directory', 'image_dir': str(root)})


if __name__ == '__main__':
    unittest.main()
