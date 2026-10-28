"""CAMO defaults, filename preservation, tuning metadata, resume, and viewing."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import generate_coco_partitions as batch
import generate_camo_partitions as camo
from view_partitions import resolve_paths, render


class CamoExportTest(unittest.TestCase):
    def test_tiny_50_defaults_export_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'camourflage_00001.jpg'
            Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(image)
            folder = 'CAMO_sam2_tiny_62_refined_50'
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({
                'data': {'name': 'CAMO', 'input_format': 'directory', 'image_dir': str(root),
                         'masks_path': str(root), 'mask_dir': 'masks', 'mask_dir_final': folder},
                'output': {'dir': str(root / 'output')},
            }))
            with patch.dict('os.environ', {'SAM2_DIR': str(root / 'sam2')}):
                with patch.object(sys, 'argv', ['generate_camo_partitions.py']):
                    defaults = camo.parse_args()
                self.assertEqual(defaults.config, 'CAMO_epito')
                self.assertEqual(defaults.model, 'sam2')
                self.assertEqual(defaults.generate_at_least_n_masks, 50)
                self.assertEqual(defaults.checkpoint, root / 'sam2/checkpoints/sam2.1_hiera_tiny.pt')
                with patch.object(sys, 'argv', ['generate_camo_partitions.py', '--config', str(config)]):
                    args = camo.parse_args()
            masks = np.ones((1, 10, 12), dtype=bool)
            history = [{'step': 1, 'n_masks': 1}]
            with patch.object(batch.sam, 'build_generator', return_value=('fake', 'cpu', 'fake.pt')) as build:
                with patch.object(batch.sam, 'infer_masks', return_value=(masks, .1, 'cpu', 'fake.pt', history)):
                    camo.run(args)
                    camo.run(args)
                    self.assertEqual(build.call_count, 1)
            output = root / 'masks' / folder
            metadata = json.loads((output / 'camourflage_00001_results.json').read_text())
            self.assertEqual(metadata['mask_tuning_history'], history)
            self.assertFalse(metadata['mask_target_met'])
            self.assertEqual(set(metadata['saved_partitions']), {'sam', 'refined'})
            self.assertIn('COMPLETE', output.with_suffix('.log').read_text())
            paths = resolve_paths(SimpleNamespace(config=str(config), image_id='camourflage_00001',
                                  image=None, partitions=None, output=None, methods=['sam', 'refined']))
            render(*paths)
            self.assertTrue(paths[2].is_file())


if __name__ == '__main__':
    unittest.main()
