"""Test CUDA retry control flow without a GPU or a real SAM installation."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import generate_sam_partitions as sam


class FakeOOM(RuntimeError):
    pass


class MemoryRetryTest(unittest.TestCase):
    def fake_torch(self):
        return SimpleNamespace(inference_mode=nullcontext, OutOfMemoryError=FakeOOM,
                               cuda=SimpleNamespace(empty_cache=Mock()))

    def test_halves_batch_retries_and_keeps_smaller_batch(self):
        torch = self.fake_torch()
        predictor = SimpleNamespace(reset_predictor=Mock())
        generator = SimpleNamespace(points_per_batch=64, predictor=predictor)
        batches = []
        def generate(image):
            batches.append(generator.points_per_batch)
            if generator.points_per_batch > 16:
                raise FakeOOM('insufficient memory')
            return [{'segmentation': np.ones(image.shape[:2], dtype=bool), 'predicted_iou': .9}]
        generator.generate = generate
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        args = SimpleNamespace(generate_at_least_n_masks=None)
        with patch.dict(sys.modules, {'torch': torch}):
            result = sam.infer_masks(image, args, (generator, 'cuda', 'fake.pt'), return_annotations=True)
            sam.infer_masks(image, args, (generator, 'cuda', 'fake.pt'))
        self.assertEqual(batches, [64, 32, 16, 16])
        self.assertEqual(result[0].shape, (1, 8, 10))
        self.assertEqual(len(result[5]), 1)
        self.assertEqual(torch.cuda.empty_cache.call_count, 2)
        self.assertEqual(predictor.reset_predictor.call_count, 2)

    def test_stops_at_one_and_preserves_tuning_thresholds(self):
        torch = self.fake_torch()
        generator = SimpleNamespace(points_per_batch=2, pred_iou_thresh=.6,
                                    stability_score_thresh=.6,
                                    generate=Mock(side_effect=FakeOOM('insufficient memory')))
        args = SimpleNamespace(generate_at_least_n_masks=50, mask_tuning_max_steps=3)
        with patch.dict(sys.modules, {'torch': torch}):
            with self.assertRaises(FakeOOM):
                sam.infer_masks(np.zeros((8, 10, 3)), args, (generator, 'cuda', 'fake.pt'))
        self.assertEqual(generator.points_per_batch, 1)
        self.assertEqual(generator.generate.call_count, 2)
        self.assertEqual(generator.pred_iou_thresh, .6)
        self.assertEqual(generator.stability_score_thresh, .6)

    def test_unrelated_errors_are_not_retried(self):
        torch = self.fake_torch()
        generator = SimpleNamespace(points_per_batch=64,
                                    generate=Mock(side_effect=ValueError('invalid input')))
        with patch.dict(sys.modules, {'torch': torch}):
            with self.assertRaisesRegex(ValueError, 'invalid input'):
                sam.infer_masks(np.zeros((8, 10, 3)), SimpleNamespace(), (generator, 'cuda', 'fake.pt'))
        self.assertEqual(generator.generate.call_count, 1)
        torch.cuda.empty_cache.assert_not_called()


if __name__ == '__main__':
    unittest.main()
