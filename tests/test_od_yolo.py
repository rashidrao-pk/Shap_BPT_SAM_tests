"""Input compatibility and unchanged detector-score behavior without model weights."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
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
            stages = ['sam', 'compact', 'refined'] if image_id.endswith('1') else ['sam', 'refined']
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
        self.assertEqual(inputs['methods'], ['BPT', 'refined'])
        self.assertEqual(inputs['output_root'], self.root / 'output/OD_yolo' / self.folder.name / 'yolo11s')

    def test_auto_uses_shared_saved_stages_and_limit(self):
        args = self.args('--methods', 'auto')
        inputs = od.prepare_inputs(self.root, self.config, args)
        self.assertEqual(inputs['methods'], ['BPT', 'sam', 'refined', 'sam_kernelshap'])
        args.limit = 1
        self.assertIn('compact', od.prepare_inputs(self.root, self.config, args)['methods'])

    def test_missing_requested_stage_and_incomplete_image_fail_before_inference(self):
        with self.assertRaisesRegex(ValueError, 'not saved'):
            od.prepare_inputs(self.root, self.config, self.args('--methods', 'boosted'))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            od.prepare_inputs(self.root, self.config, self.args('--image-ids', '3'))

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


if __name__ == '__main__':
    unittest.main()
