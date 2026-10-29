import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from check_partition_progress import inspect_folder, check_dataset, source_ids, format_bytes, format_duration


class PartitionProgressTest(unittest.TestCase):
    def test_storage_and_time_distinguish_subset_from_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for image_id, seconds in [('0', 12.5), ('1', 20)]:
                (folder / f'{image_id}_sam.npy').write_bytes(b'a' * 100)
                (folder / f'{image_id}_refined.npy').write_bytes(b'b' * 200)
                (folder / f'{image_id}_results.json').write_text(json.dumps({'total_time_sec': seconds}))
            (folder / '2_sam.npy').write_bytes(b'c' * 50)  # Partial image, no timing.
            nested = folder / 'previews'
            nested.mkdir()
            (nested / 'extra.png').write_bytes(b'd' * 80)
            row = inspect_folder(folder, {'0', '2'}, 'test')
            expected_subset = sum((folder / name).stat().st_size for name in
                                  ['0_sam.npy', '0_refined.npy', '0_results.json', '2_sam.npy'])
            self.assertEqual(row['subset_bytes'], expected_subset)
            self.assertEqual(row['folder_bytes'], sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()))
            self.assertEqual(row['subset_generation_seconds'], 12.5)
            self.assertEqual(row['folder_generation_seconds'], 32.5)
            self.assertEqual(row['subset_timed_images'], 1)
            self.assertEqual(row['folder_timed_images'], 2)
            self.assertEqual((row['completed'], row['partial']), (1, 1))

    def test_missing_or_invalid_time_is_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for image_id, value in [('0', None), ('1', -1), ('2', '10')]:
                for stage in ['sam', 'refined']:
                    (folder / f'{image_id}_{stage}.npy').touch()
                (folder / f'{image_id}_results.json').write_text(json.dumps({'total_time_sec': value}))
            row = inspect_folder(folder, {'0', '1', '2'}, 'test')
            self.assertEqual(row['completed'], 3)
            self.assertEqual(row['subset_missing_timings'], 3)
            self.assertEqual(format_duration(row['subset_generation_seconds'], row['subset_timed_images']), 'n/a')
            self.assertEqual(format_duration(3661, 1), '01:01:01')
            self.assertEqual(format_bytes(1024), '1.0 KiB')

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
