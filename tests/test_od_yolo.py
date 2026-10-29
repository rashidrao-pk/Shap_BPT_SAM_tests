"""Input compatibility and unchanged detector-score behavior without model weights."""
from contextlib import redirect_stdout
import importlib.util
import ast
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('od_yolo', ROOT / 'src/scripts/explations/OD_yolo.py')
od = importlib.util.module_from_spec(spec)
spec.loader.exec_module(od)


class ODYoloTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.folder = self.root / 'partitions/MSCOCO_sam2_large_62_refined_200'
        self.folder.mkdir(parents=True)
        self.images = self.root / 'images'
        self.images.mkdir()
        for image_id in ['000000000001', '000000000002']:
            stages = ['sam', 'compact', 'refined']
            for stage in stages:
                np.save(self.folder / f'{image_id}_{stage}.npy', np.zeros((8, 10), dtype=np.uint16))
            (self.folder / f'{image_id}_results.json').write_text(json.dumps({
                'saved_partitions': {stage: str(self.folder / f'{image_id}_{stage}.npy') for stage in stages}}))
            (self.images / f'{image_id}.jpg').touch()
        (self.folder / '000000000003_sam.npy').touch()  # Incomplete active output.
        self.annotations = self.root / 'annotations.json'
        self.annotations.write_text(json.dumps({'images': [{'id': 1}, {'id': 2}]}))
        self.config = {'data': {'name': 'MS_COCO', 'image_dir': str(self.images),
                               'annotation_file': str(self.annotations), 'masks_path': str(self.root),
                               'mask_dir': 'partitions', 'mask_dir_final': self.folder.name},
                       'evaluation': {'methods': ['BPT', 'refined']},
                       'output': {'dir': str(self.root / 'output')}}
        self.config_path = self.root / 'config.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config))

    def args(self, *extra):
        with patch.object(sys, 'argv', ['OD_yolo.py', '--config', str(self.config_path), *extra]):
            return od.parse_args()

    def test_config_selection_completed_images_and_output_isolation(self):
        inputs = od.prepare_inputs(self.root, self.config, self.args())
        self.assertEqual(inputs['image_ids'], ['000000000001', '000000000002'])
        self.assertEqual(inputs['methods'], ['BPT', 'SAM_c', 'SAM_c_kernelshap'])
        self.assertEqual(inputs['output_root'], self.root / 'output/OD_yolo' / self.folder.name / 'BPT_SAM_c_kernelshap' / 'yolo11s')

    def test_SAM_c_uses_legacy_maps_for_both_methods(self):
        self.assertEqual(od.partition_type_for_method('SAM_c'), 'compact')
        self.assertEqual(od.partition_type_for_method('SAM_c_kernelshap'), 'compact')
        inputs = od.prepare_inputs(self.root, self.config, self.args())
        self.assertEqual(inputs['method_labels']['SAM_c'], 'BPT+SAM_c')
        self.assertEqual(inputs['method_labels']['SAM_c_kernelshap'], 'SAM_c+Kernel SHAP')
        # SAM/refined maps alone must not satisfy the compact experiment.
        (self.folder / '000000000002_compact.npy').unlink()
        metadata = json.loads((self.folder / '000000000002_results.json').read_text())
        del metadata['saved_partitions']['compact']
        (self.folder / '000000000002_results.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'compact.npy'):
            od.prepare_inputs(self.root, self.config, self.args())

    def test_missing_requested_stage_and_incomplete_image_fail_before_inference(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            od.prepare_inputs(self.root, self.config, self.args('--image-ids', '3'))
        args = self.args()
        args.methods = ['boosted']
        with self.assertRaisesRegex(ValueError, 'Supported methods'):
            od.prepare_inputs(self.root, self.config, args)

    def test_dry_run_never_loads_runtime_or_creates_results(self):
        with patch.object(od, 'load_runtime_dependencies') as load:
            with redirect_stdout(io.StringIO()):
                inputs = od.run(self.args('--dry-run', '--limit', '1'))
        load.assert_not_called()
        self.assertEqual(len(inputs['image_ids']), 1)
        self.assertFalse((self.root / 'output').exists())

    def test_non_coco_config_rejected(self):
        self.config['data']['input_format'] = 'directory'
        with self.assertRaisesRegex(ValueError, 'MS-COCO'):
            od.prepare_inputs(self.root, self.config, self.args())

    def test_detector_class_aggregation(self):
        class Values:
            def __init__(self, values):
                self.values = np.array(values)
            def cpu(self):
                return self
            def numpy(self):
                return self.values
        result = SimpleNamespace(boxes=SimpleNamespace(cls=Values([0, 0, 1]), conf=Values([.3, .7, .8])))
        with patch.object(od, 'np', np, create=True):
            np.testing.assert_allclose(od.yolo_result_to_scores(result, 2, 'sum'), [1, .8])
            np.testing.assert_allclose(od.yolo_result_to_scores(result, 2, 'max'), [.7, .8])

    def test_gt_matches_category_name_and_unions_only_that_class(self):
        first = np.zeros((8, 10), dtype=bool)
        second = first.copy()
        first[:2, :2] = True
        second[1:3, 1:3] = True
        masks = {10: first, 11: second, 12: np.ones((8, 10), dtype=bool)}
        records = [{'id': 10, 'category_id': 62, 'segmentation': [1]},
                   {'id': 11, 'category_id': 62, 'segmentation': [1], 'iscrowd': 1},
                   {'id': 12, 'category_id': 1, 'segmentation': [1]}]
        class FakeCOCO:
            def getCatIds(self):
                return [1, 62]
            def loadCats(self, ids):
                return [{'id': 1, 'name': 'person'}, {'id': 62, 'name': 'chair'}]
            def getAnnIds(self, imgIds, catIds):
                return [ann['id'] for ann in records if ann['category_id'] in catIds]
            def loadAnns(self, ids):
                return [ann for ann in records if ann['id'] in ids]
            def annToMask(self, annotation):
                return masks[annotation['id']]
        gt = od.load_class_ground_truth(FakeCOCO(), 1, 'chair', (8, 10, 3))
        self.assertEqual(gt['category_ids'], [62])  # Not the YOLO zero-based index.
        self.assertEqual(gt['annotation_ids'], [10, 11])
        self.assertEqual(gt['instance_count'], 2)
        self.assertEqual(gt['foreground_pixels'], 7)
        unknown = od.load_class_ground_truth(FakeCOCO(), 1, 'custom_class', (8, 10, 3))
        self.assertFalse(unknown['mask'].any())
        self.assertEqual(unknown['status'], 'class not in COCO categories')
        # Known class with no instances is distinct from an unknown category.
        records.clear()
        absent = od.load_class_ground_truth(FakeCOCO(), 1, 'chair', (8, 10, 3))
        self.assertEqual(absent['status'], 'no matching segmentation')

    def test_gt_preview_outputs_and_resume(self):
        gt = {'mask': np.ones((8, 10), dtype=bool), 'class_name': 'chair',
              'category_ids': [62], 'annotation_ids': [10], 'instance_count': 1,
              'foreground_pixels': 80, 'status': 'matched'}
        summary = od.save_ground_truth(gt, np.zeros((8, 10, 3), dtype=np.uint8), self.root / 'gt', 1)
        self.assertNotIn('mask', summary)
        self.assertTrue(Path(summary['preview_path']).is_file())
        self.assertTrue(np.load(summary['mask_path']).all())
        self.assertEqual(json.loads((self.root / 'gt/1_gt.json').read_text())['class_name'], 'chair')
        args = self.args('--show_gt', '--no-plots')
        self.assertTrue(args.show_gt)
        (self.root / 'gt/1_predictions.json').write_text('{}')
        self.assertTrue(od.image_outputs_are_complete(self.root / 'gt', 1, [], args))
        (self.root / 'gt/1_gt.png').unlink()
        self.assertFalse(od.image_outputs_are_complete(self.root / 'gt', 1, [], args))

    def test_gt_panel_does_not_replace_explanation_panel(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tree = ast.parse((ROOT / 'src/scripts/utils_xai.py').read_text())
        plot_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'plot_xai')
        namespace = {'np': np, 'plt': plt, 'os': os, 'shapley_values_colormap': 'coolwarm',
                     'utx': SimpleNamespace(plot_predictions=lambda *args, **kwargs: None,
                                            scale_shap_values=lambda values, **kwargs: (values, 1))}
        exec(compile(ast.Module(body=[plot_function], type_ignores=[]), '<plot>', 'exec'), namespace)
        input_data = {'image_to_explain': np.zeros((8, 10, 3)), 'fixed_category': 'chair',
                      'ground_truth': {'mask': np.ones((8, 10), dtype=bool), 'class_name': 'chair',
                                       'instance_count': 1, 'status': 'matched'}}
        try:
            namespace['plot_xai'](input_data, {}, {'BPT': np.zeros((1, 8, 10))}, None, None, 1,
                                  save_path=str(self.root), save_fig=True,
                                  partition_panels=('SAM_c', 'SAM_c_kernelshap'))
            titles = [ax.get_title() for ax in plt.gcf().axes]
            self.assertTrue(any(title.startswith('GT: chair') for title in titles))
            self.assertIn('BPT Explanation', titles)
            self.assertTrue((self.root / '1_xai_results.png').is_file())
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
