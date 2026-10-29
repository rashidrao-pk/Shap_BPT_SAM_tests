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
from model_adapters import detection_scores, read_class_names


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
                self.assertTrue(importlib.import_module(kind).parse_args().save_heatmaps)
            with patch.object(sys, 'argv', [kind + '.py', '--no-heatmaps']):
                self.assertFalse(importlib.import_module(kind).parse_args().save_heatmaps)

    def test_no_heatmaps_keeps_grid_numeric_outputs_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, image_id = self.fixture(root, 'CAMO')
            args = common.create_parser('COD_ViT').parse_args([
                '--config', str(config), '--max-evals', '8', '--kernel-shap-samples', '10',
                '--target-class', '0', '--no-auc', '--no-heatmaps', '--torch-threads', '2'])
            result = common.run('COD_ViT', args, adapter=SmallAdapter())
            folder = result['run_dir'] / image_id
            self.assertTrue((folder / f'{image_id}_xai_results.png').is_file())
            self.assertTrue((folder / f'{image_id}_attributions.npz').is_file())
            manifest = json.loads((folder / f'{image_id}_complete.json').read_text())
            for method in common.METHODS:
                name = f'{method}_{image_id}.png'
                self.assertFalse((folder / name).exists())
                self.assertNotIn(name, manifest['artifacts'])
            with patch.object(common, 'load_adapter', side_effect=AssertionError('Model loaded during resume')):
                resumed = common.run('COD_ViT', args)
            self.assertEqual(resumed['completed'], 1)

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

    def test_detr_sum_max_with_sparse_coco_labels(self):
        import torch
        result = {'labels': torch.tensor([62, 62, 1]), 'scores': torch.tensor([.3, .8, .5])}
        summed = detection_scores(result, 91, 'sum')
        maximum = detection_scores(result, 91, 'max')
        self.assertAlmostEqual(summed[62], 1.1, places=6)
        self.assertAlmostEqual(maximum[62], .8, places=6)
        self.assertEqual(summed.shape, (91,))
        self.assertEqual(read_class_names({'0': 'background', '1': 'camouflage'}),
                         ['background', 'camouflage'])

    def test_custom_classifier_checkpoint_labels(self):
        import torch
        from unittest.mock import Mock
        from model_adapters import ClassifierAdapter
        args = common.create_parser('COD_ViT').parse_args(['--checkpoint', '/fixture/model.pt', '--weights', 'none'])
        checkpoint = {'state_dict': {'module.heads.head.weight': torch.zeros(2, 768)},
                      'class_names': ['background', 'camouflage']}
        model = Mock()
        with patch('torch.load', return_value=checkpoint), patch('torchvision.models.vit_b_16', return_value=model) as build:
            adapter = ClassifierAdapter(args, torch.device('cpu'), 'vit_b_16')
        self.assertEqual(adapter.class_names, ['background', 'camouflage'])
        build.assert_called_once_with(weights=None, num_classes=2)
        model.load_state_dict.assert_called_once()

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

    def test_single_region_kernel_shap_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, image_id = self.fixture(root, 'CAMO')
            np.save(next((root / 'partitions').rglob('*_compact.npy')), np.zeros((16, 20), dtype=np.uint16))
            args = common.create_parser('COD_ViT').parse_args([
                '--config', str(config), '--bg-type', 'black', '--max-evals', '8',
                '--target-class', '0', '--no-auc', '--no-plots', '--no-html', '--torch-threads', '2'])
            result = common.run('COD_ViT', args, adapter=SmallAdapter())
            with np.load(result['run_dir'] / image_id / f'{image_id}_attributions.npz') as values:
                delta = result['rows'][0]['f_S'] - result['rows'][0]['f_0']
                np.testing.assert_allclose(values['SAM_c_kernelshap'], delta)

    def test_progress_checker_recognizes_filename_stems_and_manifests(self):
        import check_experiment_progress as progress
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_id = 'camourflage_00001'
            folder = root / image_id
            folder.mkdir()
            (folder / f'{image_id}_predictions.json').write_text('{}')
            (folder / f'{image_id}_complete.json').write_text(json.dumps({
                'artifacts': [f'{image_id}_predictions.json']}))
            self.assertEqual(progress.output_status_for_image(root, image_id, common.METHODS, False), (True, []))
            (root / 'run_info.json').write_text(json.dumps({'image_ids': [image_id]}))
            self.assertEqual(progress.expected_image_ids_for_experiment(root, {'unrelated'}), {image_id})

    def test_three_row_layout_and_bpt_cut(self):
        from types import SimpleNamespace
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        tree = SimpleNamespace(N=7, U=4, cl_left=np.array([0, 2, 4]),
                               cl_right=np.array([1, 3, 5]), leaf_idx=np.arange(4),
                               cl_start=np.array([0, 2, 0]), cl_end=np.array([2, 4, 4]),
                               pixels=np.arange(4))
        labels = common.bpt_partition_labels(tree, 2, (2, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(labels[0, 0], labels[0, 1])
        maps = {method: np.ones((1, 2, 2)) for method in common.METHODS}
        curve = {'xs': [0, 1], 'y_clipr': [0, 1], 'auc_clipr': .5}
        auc = [{'label': method, 'auc_ins': curve, 'auc_del': curve} for method in common.METHODS]
        captured = []
        def capture(fig, *args, **kwargs):
            captured.extend(fig.axes[:9])
        with patch.object(Figure, 'savefig', capture):
            common.explanation_grid(SimpleNamespace(plt=plt), image, labels, maps,
                                    {'explained_class': 'object'}, 'unused.png',
                                    bpt_partition=labels, auc_results=auc)
        self.assertEqual([ax.get_title() for ax in captured[:6]],
                         ['Original', 'BPT: 2 regions', 'SAM_c: 2 regions', *common.METHODS])
        self.assertEqual([ax.get_title() for ax in captured[6:8]], ['AUC+', 'AUC−'])
        self.assertEqual(len(captured[6].lines), 3)
        self.assertFalse(captured[8].axison)

    def test_camo_gt_in_explanation_grid_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, image_id = self.fixture(root, 'CAMO')
            gt_dir = root / 'GT/Train'
            gt_dir.mkdir(parents=True)
            mask = np.zeros((16, 20), dtype=np.uint8)
            mask[:8, :10] = 255
            Image.fromarray(mask).save(gt_dir / f'{image_id}.png')
            settings = yaml.safe_load(config.read_text())
            settings['data']['ground_truth'] = {'format': 'binary', 'root': str(gt_dir.parent), 'split': 'Train'}
            config.write_text(yaml.safe_dump(settings))
            args = common.create_parser('COD_ViT').parse_args([
                '--config', str(config), '--max-evals', '8', '--kernel-shap-samples', '10',
                '--show_gt', '--no-auc', '--no-heatmaps', '--torch-threads', '2'])
            result = common.run('COD_ViT', args, adapter=SmallAdapter())
            folder = result['run_dir'] / image_id
            self.assertTrue((folder / f'{image_id}_xai_results.png').is_file())
            ground_truth = json.loads((folder / f'{image_id}_gt.json').read_text())
            self.assertEqual(ground_truth['class_name'], 'Camouflaged object')
            self.assertEqual(ground_truth['scope'], 'dataset_foreground')
            self.assertEqual(ground_truth['foreground_pixels'], 80)
            manifest = json.loads((folder / f'{image_id}_complete.json').read_text())
            self.assertIn(f'{image_id}_gt_labels.npz', manifest['artifacts'])


if __name__ == '__main__':
    unittest.main()
