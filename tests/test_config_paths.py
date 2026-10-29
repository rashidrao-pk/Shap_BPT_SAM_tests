"""Check the three-root config contract and consumers using only synthetic data."""
import json
import ast
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
from config_paths import output_path, partition_path, evaluation_path
import generate_coco_partitions as batch
from check_partition_progress import check_dataset
from view_partitions import resolve_paths, render
import check_experiment_progress as experiment_progress


class ConfigPathsTest(unittest.TestCase):
    def test_new_roots_take_precedence_and_relative_paths_are_rooted(self):
        config = {'data': {'masks_path': '/old', 'mask_dir': 'old', 'mask_dir_final': 'dataset'},
                  'output': {'dir': '/old', 'folder': 'results', 'partitions': 'p',
                             'explanations': 'x', 'evaluations': 'e'}}
        root = Path('/project')
        self.assertEqual(partition_path(config, root), root / 'p/dataset')
        run = root / 'x/OD_yolo/dataset/bg_black_B500'
        self.assertEqual(evaluation_path(config, run, root), root / 'e/OD_yolo/dataset/bg_black_B500')
        self.assertEqual(evaluation_path(config, Path('/custom/bg_black_B500'), root), root / 'e/bg_black_B500')

    def test_old_roots_remain_readable(self):
        config = {'data': {'masks_path': '/datasets', 'mask_dir': 'SAM_BPT', 'mask_dir_final': 'old'},
                  'output': {'dir': '/project/src', 'folder': '/project/outputs/results'}}
        self.assertEqual(partition_path(config), Path('/datasets/SAM_BPT/old'))
        self.assertEqual(output_path(config, 'explanations'), Path('/project/outputs/results'))
        with self.assertRaisesRegex(ValueError, 'output.partitions'):
            output_path({}, 'partitions')

    def test_generation_progress_and_preview_use_separate_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images'
            images.mkdir()
            Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(images / '0.jpg')
            name = 'CAMO_sam2_tiny_62_refined_singlepass'
            config = {'data': {'name': 'CAMO', 'input_format': 'directory',
                               'image_dir': str(images), 'mask_dir_final': name},
                      'output': {'partitions': str(root / 'partitions'),
                                 'explanations': str(root / 'explanations'),
                                 'evaluations': str(root / 'evaluations')}}
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            args = batch.create_parser().parse_args(['--config', str(path)])
            masks = np.ones((1, 8, 10), dtype=bool)
            with patch.object(batch.sam, 'build_generator', return_value=('fake', 'cpu', 'fake.pt')):
                with patch.object(batch.sam, 'infer_masks', return_value=(masks, .1, 'cpu', 'fake.pt')):
                    batch.run(args)
            self.assertTrue((root / 'partitions' / name / '0_refined.npy').is_file())
            self.assertTrue((root / 'partitions' / f'{name}.log').is_file())
            self.assertEqual(check_dataset(str(path), configured_only=True)[0]['completed'], 1)
            image, partitions, output = resolve_paths(SimpleNamespace(
                config=str(path), image_id='0', image=None, partitions=None, output=None,
                methods=['sam', 'refined']))
            self.assertEqual(output, root / 'evaluations/previews' / name / '0.png')
            render(image, partitions, output)
            self.assertTrue(output.is_file())
            self.assertFalse((root / 'explanations').exists())

    def test_all_repository_configs_define_the_new_schema(self):
        for path in (ROOT / 'src/configs').glob('*.yaml'):
            config = yaml.safe_load(path.read_text())
            with self.subTest(config=path.name):
                self.assertTrue(all(config['output'].get(key) for key in
                                    ['partitions', 'explanations', 'evaluations']))
                self.assertNotIn('dir', config['output'])
                self.assertNotIn('folder', config['output'])
                self.assertNotIn('masks_path', config.get('data', {}))
                self.assertNotIn('mask_dir', config.get('data', {}))

    def test_evaluation_tools_resolve_new_roots_and_nested_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {'data': {'mask_dir_final': 'dataset'},
                      'output': {'partitions': str(root / 'p'), 'explanations': str(root / 'x'),
                                 'evaluations': str(root / 'e')}}
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            args = SimpleNamespace(config=str(path), results_root=None, masks_base_path=None,
                                   results_dir=None, bg_type='black', max_evals=500)
            self.assertEqual(experiment_progress.resolve_results_root(root, args), root / 'x')
            self.assertEqual(experiment_progress.resolve_masks_base_path(root, args), root / 'p/dataset')
            run = root / 'x/OD_yolo/dataset/bg_black_B500'
            run.mkdir(parents=True)
            (run / 'run_info.json').write_text('{}')
            self.assertEqual(experiment_progress.discover_experiment_dirs(root / 'x', None), [run])
            # Test the summary input resolver without importing detector dependencies.
            source = ast.parse((ROOT / 'src/scripts/summarize_results.py').read_text())
            function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                            and node.name == 'resolve_results_dir')
            namespace = {'Path': Path, 'output_path': output_path,
                         'load_config': lambda project_root, name: config}
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<summary paths>', 'exec'), namespace)
            self.assertEqual(namespace['resolve_results_dir'](root, args), run)


if __name__ == '__main__':
    unittest.main()
