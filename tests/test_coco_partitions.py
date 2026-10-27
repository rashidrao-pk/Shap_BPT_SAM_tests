"""Exercise the dataset export/loader contract without a GPU or checkpoints."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / 'src/scripts'
sys.path.insert(0, str(SCRIPTS))
import generate_coco_partitions as batch
import utils_sam_2 as loader


class CocoExportTest(unittest.TestCase):
    def test_export_resume_and_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            records = []
            for number in [2, 1]:
                name = f'{number:012d}.jpg'
                Image.fromarray(image).save(root / name)
                records.append({'id': number, 'file_name': name})
            (root / 'annotations.json').write_text(json.dumps({'images': records}))
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'data': {
                'image_dir': str(root), 'annotation_file': str(root / 'annotations.json'),
                'masks_path': str(root), 'mask_dir': 'masks', 'mask_dir_final': 'sam2',
            }}))
            args = argparse.Namespace(config=str(config), model='sam2', checkpoint=None,
                                      max_regions=3, min_region_pixels=1, limit=1,
                                      overwrite=False, dry_run=False)
            masks = np.zeros((2, 8, 10), dtype=bool)
            masks[0, :5, :5] = True
            masks[1, 4:, 4:] = True
            with patch.object(batch.sam, 'build_generator', return_value=('fake', 'cpu', 'fake.pt')) as build:
                with patch.object(batch.sam, 'infer_masks', return_value=(masks, 0.1, 'cpu', 'fake.pt')):
                    batch.run(args)
                    output = root / 'masks/sam2'
                    for method in ['sam', 'refined']:
                        original, capped = loader.load_refine_partitions(image, str(output), '000000000001', method)
                        self.assertEqual(original.shape, image.shape[:2])
                        self.assertLessEqual(len(np.unique(original)), 3)
                        self.assertEqual(capped.shape, original.shape)
                    self.assertFalse((output / '000000000002_refined.npy').exists())
                    self.assertEqual(loader.load_json_partitions(str(output), '000000000001')['model'], 'sam2')
                    batch.run(args)
                    self.assertEqual(build.call_count, 1)
                    args.limit = None
                    batch.run(args)
                    self.assertTrue((output / '000000000002_refined.npy').exists())
                    self.assertEqual(build.call_count, 2)
                    args.max_regions = 4
                    with self.assertRaisesRegex(ValueError, 'settings differ'):
                        batch.run(args)

    def test_empty_masks(self):
        masks = np.zeros((0, 5, 6), dtype=bool)
        labels = batch.sam.refined_partitions(masks, 1, 3)
        self.assertEqual(labels.shape, (5, 6))
        self.assertEqual(len(np.unique(labels)), 1)


if __name__ == '__main__':
    unittest.main()
