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


class NewDatasetProgressTest(unittest.TestCase):
    def test_new_datasets_are_default(self):
        from check_partition_progress import DEFAULT_CONFIGS
        self.assertIn('BUSI_epito', DEFAULT_CONFIGS)
        self.assertIn('PartImageNet_epito', DEFAULT_CONFIGS)

    def test_busi_excludes_ground_truth_and_counts_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images/benign'
            images.mkdir(parents=True)
            for name in ['benign (1).png', 'benign (1)_mask.png', 'benign (1)_mask_1.png', 'benign (2).png']:
                (images / name).touch()
            self.assertEqual(source_ids({'name': 'BUSI', 'image_dir': str(images.parent)}, 1000),
                             {'benign (1)', 'benign (2)'})
            base = root / 'partitions'
            folder = base / 'BUSI_sam2_tiny_62_compact-refined-sam_50'
            folder.mkdir(parents=True)
            (folder / 'benign (1)_compact.npy').write_bytes(b'map')
            (folder / 'benign (1)_results.json').write_text(json.dumps({
                'saved_partitions': {'compact': 'path'}, 'total_time_sec': 12}))
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'data': {'name': 'BUSI', 'input_format': 'directory',
                'image_dir': str(images.parent), 'mask_dir_final': folder.name},
                'output': {'partitions': str(base)}}))
            row = check_dataset(str(config), limit=1000)[0]
            self.assertEqual((row['completed'], row['total'], row['remaining']), (1, 2, 1))
            self.assertEqual(row['subset_generation_seconds'], 12)

    def test_partimagenet_uses_selected_annotation_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images/val'
            images.mkdir(parents=True)
            for name in ['n1_1.JPEG', 'n2_2.JPEG', 'unannotated.JPEG']:
                (images / name).touch()
            annotation = root / 'val.json'
            annotation.write_text(json.dumps({'images': [
                {'id': 7, 'file_name': 'n2_2.JPEG'}, {'id': 4, 'file_name': 'n1_1.JPEG'}]}))
            data = {'name': 'PartImageNet', 'input_format': 'directory',
                    'image_dir': str(images.parent), 'annotation_file': str(annotation)}
            self.assertEqual(source_ids(data, 1), {'n1_1'})
            self.assertEqual(source_ids(data, 1000), {'n1_1', 'n2_2'})

    def test_nested_busi_partition_outputs_exclude_xai_storage(self):
        from check_partition_progress import inspect_busi_pipeline
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images/benign'
            images.mkdir(parents=True)
            for image_id in ['benign (1)', 'benign (2)']:
                (images / (image_id + '.png')).touch()
            results = root / 'results'
            complete = results / 'benign_benign_(1)/sam1'
            complete.mkdir(parents=True)
            (complete / 'input_sam1_refined_62.npy').write_bytes(b'partition')
            metadata = complete / 'input_sam1_refined_62.json'
            metadata.write_text(json.dumps({'inference_seconds': 3.5, 'refinement_seconds': 1.5}))
            (complete / 'attribution.npy').write_bytes(b'x' * 200)
            partial = results / 'benign_benign_(2)/sam2'
            partial.mkdir(parents=True)
            (partial / 'input_sam2_masks.npz').write_bytes(b'raw')
            rows = inspect_busi_pipeline(results, {'image_dir': str(images.parent)}, {'benign (1)', 'benign (2)'})
            self.assertEqual(len(rows), 2)
            self.assertEqual((rows[0]['completed'], rows[0]['total']), (1, 2))
            self.assertEqual(rows[0]['subset_generation_seconds'], 5)
            self.assertEqual(rows[0]['subset_bytes'], len(b'partition') + metadata.stat().st_size)
            self.assertEqual((rows[1]['completed'], rows[1]['partial']), (0, 1))

    def test_busi_summary_needs_no_scientific_imports(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images/benign'
            images.mkdir(parents=True)
            (images / 'benign (1).png').touch()
            (images / 'benign (1)_mask.png').touch()
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'data': {'name': 'BUSI', 'image_dir': str(images.parent),
                'mask_dir_final': 'BUSI_test'}, 'output': {'partitions': str(root / 'partitions')}}))
            harness = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'PIL', 'torch'}:
            raise AssertionError('Unexpected scientific import: ' + fullname)
sys.meta_path.insert(0, Block())
script, config = sys.argv[1:]
sys.path.insert(0, str(__import__('pathlib').Path(script).parent))
sys.argv = [script, '--config', config, '--configured-only', '--limit', '1000']
runpy.run_path(script, run_name='__main__')
'''
            script = Path(__file__).resolve().parents[1] / 'src/scripts/check_partition_progress.py'
            result = subprocess.run([sys.executable, '-c', harness, str(script), str(config)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('0 / 1', result.stdout)

    def test_imagenet_gt_only_filters_before_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images'
            images.mkdir()
            ids = [f'ILSVRC2012_val_{n:08d}' for n in range(1, 7)]
            for image_id in ids:
                (images / (image_id + '.JPEG')).touch()
            gt = root / 'GT/ImageNetS50/validation-segmentation/n00000001'
            gt.mkdir(parents=True)
            # Annotated inputs are beyond the first two unfiltered images.
            for image_id in [*ids[-2:], 'ILSVRC2012_val_99999999']:
                (gt / (image_id + '.png')).touch()
            base = root / 'partitions'
            name = 'ImageNet_sam2_tiny_62_compact-sam_100'
            folder = base / name
            folder.mkdir(parents=True)
            for image_id in ids[-2:]:
                for stage in ['sam', 'compact']:
                    (folder / f'{image_id}_{stage}.npy').touch()
                (folder / f'{image_id}_results.json').write_text(json.dumps({
                    'saved_partitions': {'sam': '', 'compact': ''}, 'total_time_sec': 5}))
            data = {'name': 'ImageNet', 'input_format': 'directory', 'image_dir': str(images),
                    'mask_dir_final': name, 'ground_truth': {'format': 'imagenets',
                    'root': str(root / 'GT'), 'variant': 50}}
            self.assertEqual(source_ids(data, 2), set(ids[:2]))
            self.assertEqual(source_ids(data, 1, gt_only=True), {ids[-2]})
            self.assertEqual(source_ids(data, 1000, gt_only=True), set(ids[-2:]))
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'data': data, 'output': {'partitions': str(base)}}))
            plain = check_dataset(str(config), limit=2)[0]
            selected = check_dataset(str(config), limit=1000, gt_only=True)[0]
            self.assertEqual((plain['completed'], plain['total']), (0, 2))
            self.assertEqual((selected['completed'], selected['total'], selected['percent']), (2, 2, 100))
            self.assertEqual(selected['subset_generation_seconds'], 10)
            self.assertEqual(selected['input_selection'], 'gt-only')
            self.assertEqual(selected['gt_variant'], 50)
            # A global --gt-only does not change other datasets' source selection.
            self.assertEqual(source_ids({**data, 'name': 'CAMO'}, 2, gt_only=True), set(ids[:2]))

    def test_imagenet_gt_only_cli_without_numpy(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images'
            images.mkdir()
            (images / 'ILSVRC2012_val_00000001.JPEG').touch()
            (images / 'ILSVRC2012_val_00000099.JPEG').touch()
            gt = root / 'ImageNetS50/validation-segmentation/n00000001'
            gt.mkdir(parents=True)
            (gt / 'ILSVRC2012_val_00000099.png').touch()
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'data': {'name': 'ImageNet', 'input_format': 'directory',
                'image_dir': str(images), 'mask_dir_final': 'ImageNet_test',
                'ground_truth': {'format': 'imagenets', 'root': str(root / 'ImageNetS50'), 'variant': 50}},
                'output': {'partitions': str(root / 'partitions')}}))
            harness = '''
import importlib.abc, runpy, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'PIL', 'scipy'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
script, config = sys.argv[1:]
sys.path.insert(0, str(Path(script).parent))
sys.argv = [script, '--config', config, '--gt-only', '--limit', '1']
runpy.run_path(script, run_name='__main__')
'''
            script = Path(__file__).resolve().parents[1] / 'src/scripts/check_partition_progress.py'
            result = subprocess.run([sys.executable, '-c', harness, str(script), str(config)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('0 / 1', result.stdout)
            self.assertIn('GT-only selection', result.stdout)
