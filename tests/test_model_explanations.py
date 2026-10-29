"""Exercise dataset routing, real ShapBPT/Kernel SHAP, reports, and resume."""
import contextlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts/explanations'))
import experiment_common as common
from model_adapters import detection_scores


class SmallAdapter:
    class_names = ['object', 'other']
    def predict_batch(self, images):
        values = np.asarray([image[:, :, 0].mean() / 255 for image in images])
        return np.column_stack([values, 1-values])
    def detections(self, image):
        return []


class ModelExplanationTest(unittest.TestCase):
    def fixture(self, root, dataset):
        images = root / 'images'
        images.mkdir()
        image_id = {'MS_COCO': '000000000139', 'ImageNet': 'ILSVRC2012_val_00000001',
                    'CAMO': 'camourflage_00001'}[dataset]
        image = np.zeros((16, 20, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(20)[None, :] * 12
        image[:, :, 1] = 80
        image[:, :, 2] = 120
        extension = '.JPEG' if dataset == 'ImageNet' else '.jpg'
        Image.fromarray(image).save(images / f'{image_id}{extension}')
        partition_dir = root / 'partitions' / f'{dataset}_sam2_tiny_62_refined_50'
        partition_dir.mkdir(parents=True)
        labels = np.zeros((16, 20), dtype=np.uint16)
        labels[:8, 10:] = 1
        labels[8:, :10] = 2
        labels[8:, 10:] = 3
        np.save(partition_dir / f'{image_id}_compact.npy', labels)
        (partition_dir / f'{image_id}_results.json').write_text(json.dumps({
            'saved_partitions': {'compact': str(partition_dir / f'{image_id}_compact.npy')}}))
        data = {'name': dataset, 'image_dir': str(images), 'mask_dir_final': partition_dir.name}
        if dataset == 'MS_COCO':
            annotation_file = root / 'annotations.json'
            annotation_file.write_text(json.dumps({'images': [{'id': 139, 'file_name': f'{image_id}{extension}',
                                                               'height': 16, 'width': 20}],
                                                    'annotations': [], 'categories': []}))
            data['annotation_file'] = str(annotation_file)
        else:
            data['input_format'] = 'directory'
        path = root / 'config.yaml'
        path.write_text(yaml.safe_dump({'data': data, 'device': 'cpu',
                                      'output': {'partitions': str(root / 'partitions'),
                                                 'explanations': str(root / 'explanations'),
                                                 'evaluations': str(root / 'evaluations')}}))
        return path, image_id

    def test_entrypoint_defaults(self):
        for kind, config in common.DEFAULT_CONFIGS.items():
            with patch.object(sys, 'argv', [kind + '.py']):
                self.assertEqual(importlib.import_module(kind).parse_args().config, config)

    def test_preflight_routes_datasets_without_loading_models(self):
        for kind, dataset in [('OD_DETR', 'MS_COCO'), ('OC_ResNet', 'ImageNet'),
                              ('OC_ViT', 'ImageNet'), ('COD_ViT', 'CAMO')]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, image_id = self.fixture(root, dataset)
                args = common.create_parser(kind).parse_args(['--config', str(config), '--dry-run'])
                with patch.object(common, 'load_runtime') as runtime, contextlib.redirect_stdout(None):
                    result = common.run(kind, args)
                runtime.assert_not_called()
                self.assertEqual(result['records'][0]['image_id'], image_id)
                self.assertEqual(result['methods'], ['BPT', 'SAM_c', 'SAM_c_kernelshap'])
                self.assertIn(kind, str(result['run_dir']))
                self.assertFalse((root / 'explanations').exists())

    def test_mask_background_averaging(self):
        image = np.full((4, 5, 3), 255, dtype=np.uint8)
        predict = common.make_masked_predictor(SmallAdapter(), image,
                                              [np.zeros_like(image), np.full_like(image, 127)], 2)
        masks = np.stack([np.ones((4, 5), dtype=bool), np.zeros((4, 5), dtype=bool)])
        scores = predict(masks)
        self.assertAlmostEqual(scores[0, 0], 1)
        self.assertAlmostEqual(scores[1, 0], 127/510)
        self.assertEqual(predict(np.empty((0, 4, 5), dtype=bool)).shape, (0, 2))

    def test_real_explanations_metrics_reports_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, image_id = self.fixture(root, 'ImageNet')
            args = common.create_parser('OC_ResNet').parse_args([
                '--config', str(config), '--bg-type', 'black', '--max-evals', '8',
                '--kernel-shap-samples', '10', '--auc-num-samples', '5',
                '--target-class', '0', '--torch-threads', '2', '--verbose-level', 'low'])
            result = common.run('OC_ResNet', args, adapter=SmallAdapter())
            self.assertEqual(result['completed'], 1)
            self.assertEqual([row['method'] for row in result['rows']], common.METHODS)
            for row in result['rows']:
                self.assertTrue(np.isfinite(row['auc_ins']))
                self.assertTrue(np.isfinite(row['auc_del']))
            folder = result['run_dir'] / image_id
            self.assertTrue((folder / f'{image_id}_xai_results.png').is_file())
            values = np.load(folder / f'{image_id}_attributions.npz')
            self.assertEqual(set(values.files), set(common.METHODS))
            for method in common.METHODS:
                self.assertEqual(values[method].shape, (1, 16, 20))
            self.assertTrue((result['evaluation_dir'] / 'method_summary.csv').is_file())
            report = result['evaluation_dir'] / 'xai_results_report.html'
            self.assertIn(image_id, report.read_text())
            with patch.object(common, 'load_adapter', side_effect=AssertionError('Model loaded during resume')):
                resumed = common.run('OC_ResNet', args)
            self.assertEqual(resumed['completed'], 1)
            self.assertEqual(len(resumed['rows']), 3)
            args.target_class = 1
            with self.assertRaisesRegex(ValueError, 'settings/inputs changed'):
                common.run('OC_ResNet', args)

    def test_missing_compact_and_wrong_dataset_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, image_id = self.fixture(root, 'CAMO')
            args = common.create_parser('OC_ViT').parse_args(['--config', str(config)])
            with self.assertRaisesRegex(ValueError, 'imagenet'):
                common.prepare_inputs('OC_ViT', args)
            args = common.create_parser('COD_ViT').parse_args(['--config', str(config)])
            next((root / 'partitions').rglob('*_compact.npy')).unlink()
            with self.assertRaisesRegex(ValueError, 'No completed SAM_c'):
                common.prepare_inputs('COD_ViT', args)


if __name__ == '__main__':
    unittest.main()
