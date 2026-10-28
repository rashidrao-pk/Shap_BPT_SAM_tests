import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from check_partition_progress import inspect_folder, check_dataset, source_ids


class PartitionProgressTest(unittest.TestCase):
    def test_complete_partial_and_recorded_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ['0_sam.npy', '0_refined.npy', '1_boosted.npy', '2_sam.npy']:
                (folder / name).touch()
            (folder / '0_results.json').write_text('{}')
            (folder / '1_results.json').write_text(json.dumps({'saved_partitions': {'boosted': '/old/path'}}))
            (folder / '2_results.json').write_text('invalid')
            row = inspect_folder(folder, {'0', '1', '2', '3'}, 'test')
            self.assertEqual((row['completed'], row['partial'], row['remaining']), (2, 1, 2))
            self.assertEqual(row['invalid_metadata'], 1)
            self.assertEqual(row['percent'], 50)
            required = inspect_folder(folder, {'0', '1', '2', '3'}, 'test', ['sam', 'refined'])
            self.assertEqual(required['completed'], 1)

    def test_dataset_variants_and_missing_configured_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images'
            images.mkdir()
            (images / 'camourflage_00001.jpg').touch()
            (images / 'camourflage_00002.jpg').touch()
            (images / 'not_image.txt').touch()
            variant = root / 'masks/CAMO_sam2_tiny_62_refined_50'
            variant.mkdir(parents=True)
            for suffix in ['sam.npy', 'refined.npy']:
                (variant / f'camourflage_00001_{suffix}').touch()
            (variant / 'camourflage_00001_results.json').write_text('{}')
            data = {'name': 'CAMO', 'input_format': 'directory', 'image_dir': str(images),
                    'masks_path': str(root), 'mask_dir': 'masks', 'mask_dir_final': 'CAMO_sam2_large_62_refined_50'}
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'data': data}))
            rows = check_dataset(str(config))
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(row['completed'] for row in rows), 1)
            self.assertEqual({row['total'] for row in rows}, {2})
            self.assertEqual(check_dataset(str(config), configured_only=True)[0]['status'], 'not created')
            self.assertEqual(source_ids(data, 1), {'camourflage_00001'})

    def test_coco_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'annotations.json'
            path.write_text(json.dumps({'images': [{'id': 139}, {'id': 1}]}))
            self.assertEqual(source_ids({'annotation_file': str(path)}, 1), {'000000000001'})


if __name__ == '__main__':
    unittest.main()
