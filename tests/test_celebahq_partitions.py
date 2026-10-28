"""Check CelebA-HQ filename preservation, export, resume, logs, and preview paths."""
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
import generate_celebahq_partitions as celeb
from view_partitions import resolve_paths, render


class CelebAHQExportTest(unittest.TestCase):
    def test_export_resume_and_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'images'
            images.mkdir()
            rgb = np.zeros((10, 12, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(images / '0.jpg')
            Image.fromarray(rgb).save(images / '1.jpg')
            config = root / 'config.yaml'
            folder = 'CelebAHQ_sam2_tiny_62_refined_singlepass'
            config.write_text(yaml.safe_dump({
                'data': {'name': 'CelebAHQ', 'input_format': 'directory',
                         'image_dir': str(images), 'masks_path': str(root),
                         'mask_dir': 'masks', 'mask_dir_final': folder},
                'output': {'dir': str(root / 'output')},
            }))
            with patch.object(sys, 'argv', ['generate_celebahq_partitions.py']):
                self.assertEqual(celeb.parse_args().config, 'CelebAHQ_epito')
            with patch.object(sys, 'argv', ['generate_celebahq_partitions.py', '--config', str(config), '--limit', '1']):
                args = celeb.parse_args()
            masks = np.ones((1, 10, 12), dtype=bool)
            with patch.object(batch.sam, 'build_generator', return_value=('fake', 'cpu', 'fake.pt')) as build:
                with patch.object(batch.sam, 'infer_masks', return_value=(masks, .1, 'cpu', 'fake.pt')):
                    celeb.run(args)
                    celeb.run(args)
                    self.assertEqual(build.call_count, 1)
            output = root / 'masks' / folder
            self.assertTrue((output / '0_sam.npy').is_file())
            self.assertFalse((output / '1_sam.npy').exists())
            metadata = json.loads((output / '0_results.json').read_text())
            self.assertEqual(metadata['source_image'], str(images / '0.jpg'))
            self.assertEqual(set(metadata['saved_partitions']), {'sam', 'refined'})
            self.assertIn('COMPLETE', output.with_suffix('.log').read_text())
            paths = resolve_paths(SimpleNamespace(config=str(config), image_id='0',
                                  image=None, partitions=None, output=None,
                                  methods=['sam', 'refined']))
            self.assertEqual(paths[0].name, '0.jpg')
            render(*paths)
            self.assertTrue(paths[2].is_file())


if __name__ == '__main__':
    unittest.main()
