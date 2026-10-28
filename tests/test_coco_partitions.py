"""Exercise the dataset export/loader contract without a GPU or checkpoints."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / 'src/scripts'
sys.path.insert(0, str(SCRIPTS))
import generate_coco_partitions as batch
import utils_sam_2 as loader


class CocoExportTest(unittest.TestCase):
    def test_selected_notebook_stages_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb = np.zeros((8, 10, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(root / '000000000001.jpg')
            (root / 'annotations.json').write_text(json.dumps({'images': [
                {'id': 1, 'file_name': '000000000001.jpg'}]}))
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'data': {
                'image_dir': str(root), 'annotation_file': str(root / 'annotations.json'),
                'masks_path': str(root), 'mask_dir': 'masks'}}))
            args = batch.create_parser().parse_args([
                '--config', str(config), '--save-partitions', 'compact', 'boosted',
                'filled', 'boosted_filled'])
            masks = np.zeros((2, 8, 10), dtype=bool)
            masks[0, :5, :5] = True
            masks[1, 4:, 4:] = True
            annotations = [{'segmentation': mask, 'area': int(mask.sum()), 'predicted_iou': 0.9}
                           for mask in masks]
            with patch.object(batch.sam, 'build_generator', return_value=('fake', 'cpu', 'fake.pt')) as build:
                with patch.object(batch.sam, 'infer_masks', return_value=(masks, .1, 'cpu', 'fake.pt', [], annotations)):
                    batch.run(args)
                    batch.run(args)
                    self.assertEqual(build.call_count, 1)
            output = root / 'masks/MSCOCO_sam2_tiny_62_refined_singlepass'
            self.assertFalse((output / '000000000001_sam.npy').exists())
            self.assertFalse((output / '000000000001_refined.npy').exists())
            metadata = json.loads((output / '000000000001_results.json').read_text())
            self.assertEqual(set(metadata['saved_partitions']), set(args.save_partitions))
            for name in args.save_partitions:
                labels = np.load(metadata['saved_partitions'][name])
                self.assertEqual(labels.shape, rgb.shape[:2])
                self.assertEqual(len(np.unique(labels)), metadata['partition_label_counts'][name])
            self.assertEqual(metadata['n_coverage_masks'], 2)
            zero_maps, _ = batch.notebook_stage_maps([], rgb.shape[:2], args.save_partitions)
            for labels in zero_maps.values():
                self.assertFalse(labels.any())

    def test_mask_tuning_stops_and_resets_between_images(self):
        thresholds = []
        generator = SimpleNamespace(pred_iou_thresh=0.6, stability_score_thresh=0.6)
        def generate(image):
            thresholds.append((generator.pred_iou_thresh, generator.stability_score_thresh))
            return [None] * (5 if generator.pred_iou_thresh < 0.5 else 1)
        generator.generate = generate
        for _ in range(2):
            masks, history = batch.sam.generate_at_least_n_masks(None, generator, min_masks=5)
            self.assertEqual(len(masks), 5)
            self.assertEqual(len(history), 2)
            self.assertEqual(generator.pred_iou_thresh, 0.6)
        self.assertEqual(len(thresholds), 4)
        self.assertEqual(thresholds[0], thresholds[2])

    def test_mask_tuning_bounds_and_exception_reset(self):
        generator = SimpleNamespace(pred_iou_thresh=0.6, stability_score_thresh=0.6,
                                    generate=lambda image: [])
        masks, history = batch.sam.generate_at_least_n_masks(None, generator, max_steps=10)
        self.assertEqual(masks, [])
        self.assertEqual(len(history), 4)
        self.assertEqual(history[-1]['pred_iou_thresh'], 0.1)
        _, history = batch.sam.generate_at_least_n_masks(None, generator, max_steps=3)
        self.assertEqual(len(history), 3)
        with patch.object(generator, 'generate', side_effect=RuntimeError('inference failed')):
            with self.assertRaises(RuntimeError):
                batch.sam.generate_at_least_n_masks(None, generator)
        self.assertEqual(generator.pred_iou_thresh, 0.6)
        self.assertEqual(generator.stability_score_thresh, 0.6)

    def test_export_resume_and_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((8, 10, 3), dtype=np.uint8)
            records = []
            for number in [2, 1]:
                name = f'{number:012d}.jpg'
                Image.fromarray(image).save(root / name)
                records.append({'id': number, 'file_name': name})
            (root / 'annotations.json').write_text(json.dumps({'images': records}))
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'data': {
                'image_dir': str(root), 'annotation_file': str(root / 'annotations.json'),
                'masks_path': str(root), 'mask_dir': 'masks', 'mask_dir_final': 'sam2',
            }}))
            args = argparse.Namespace(config=str(config), model='sam2', checkpoint=None,
                                      max_regions=3, min_region_pixels=1, limit=1,
                                      overwrite=False, dry_run=False)
            masks = np.zeros((2, 8, 10), dtype=bool)
            masks[0, :5, :5] = True
            masks[1, 4:, 4:] = True
            with patch.object(batch.sam, 'build_generator', return_value=('fake', 'cpu', 'fake.pt')) as build:
                with patch.object(batch.sam, 'infer_masks', return_value=(masks, 0.1, 'cpu', 'fake.pt')):
                    batch.run(args)
                    output = root / 'masks/MSCOCO_sam2_tiny_3_refined_singlepass'
                    for method in ['sam', 'refined']:
                        original, capped = loader.load_refine_partitions(image, str(output), '000000000001', method)
                        self.assertEqual(original.shape, image.shape[:2])
                        self.assertLessEqual(len(np.unique(original)), 3)
                        self.assertEqual(capped.shape, original.shape)
                    self.assertFalse((output / '000000000002_refined.npy').exists())
                    self.assertEqual(loader.load_json_partitions(str(output), '000000000001')['model'], 'sam2')
                    log = (root / 'masks/MSCOCO_sam2_tiny_3_refined_singlepass.log').read_text()
                    self.assertIn('000000000001: 2 masks, 2 regions', log)
                    self.assertIn('sam_s=0.100', log)
                    self.assertIn('refinement_s=', log)
                    self.assertIn('COMPLETE', log)
                    self.assertNotIn('it/s', log)
                    args.log_file = root / 'other.log'
                    batch.run(args)
                    self.assertEqual(build.call_count, 1)
                    self.assertIn('COMPLETE', args.log_file.read_text())
                    args.limit = None
                    batch.run(args)
                    self.assertTrue((output / '000000000002_refined.npy').exists())
                    self.assertEqual(build.call_count, 2)
                    args.min_region_pixels = 2
                    with self.assertRaisesRegex(ValueError, 'settings differ'):
                        batch.run(args)
                    args.min_region_pixels = 1
                    args.max_regions = 3
                    args.overwrite = True
                    args.generate_at_least_n_masks = 5
                    args.mask_tuning_max_steps = 3
                    history = [{'step': 1, 'n_masks': 2}]
                    with patch.object(batch.sam, 'infer_masks',
                                      return_value=(masks, 0.1, 'cpu', 'fake.pt', history)):
                        batch.run(args)
                    output = root / 'masks/MSCOCO_sam2_tiny_3_refined_5'
                    for image_id in ['000000000001', '000000000002']:
                        metadata = loader.load_json_partitions(str(output), image_id)
                        self.assertFalse(metadata['mask_target_met'])
                        self.assertEqual(metadata['mask_tuning_history'], history)

    def test_folder_names(self):
        args = argparse.Namespace(model='sam2', checkpoint=Path('sam2.1_hiera_tiny.pt'),
                                  max_regions=62, generate_at_least_n_masks=200)
        self.assertEqual(batch.partition_folder_name({'name': 'MS_COCO'}, args),
                         'MSCOCO_sam2_tiny_62_refined_200')
        args.checkpoint = Path('sam2.1_hiera_large.pt')
        self.assertEqual(batch.partition_folder_name({'name': 'MS_COCO'}, args),
                         'MSCOCO_sam2_large_62_refined_200')
        args.model = 'sam1'
        args.sam1_variant = 'vit_b'
        self.assertEqual(batch.partition_folder_name({'name': 'MS_COCO'}, args),
                         'MSCOCO_sam1_vit_b_62_refined_200')

    def test_empty_masks(self):
        masks = np.zeros((0, 5, 6), dtype=bool)
        labels = batch.sam.refined_partitions(masks, 1, 3)
        self.assertEqual(labels.shape, (5, 6))
        self.assertEqual(len(np.unique(labels)), 1)


if __name__ == '__main__':
    unittest.main()
