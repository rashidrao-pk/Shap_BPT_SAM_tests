"""Baseline algorithms, geometry, truthful unsupported methods, and GT IoU."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'))
import all_methods as allx
import experiment_common as common


class AllMethodsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def args(self):
        return common.create_parser('OC_ResNet').parse_args([
            '--explain_all', '--bpt-budgets', '8', '--lime-budgets', '8',
            '--gradient-samples', '2', '--idg-steps', '4', '--attribution-batch-size', '2'])

    def test_iou_thresholds_respect_ties_and_ignore(self):
        gt = {'mask': np.array([[1, 1], [0, 0]], dtype=bool),
              'label_map': np.array([[1, 1], [0, 1000]])}
        result = allx.best_iou(np.array([[.9, .9], [.1, 9.0]]), gt)
        self.assertEqual(result['best_iou'], 1.)
        self.assertEqual(result['threshold'], .9)
        self.assertFalse(result['mask'][1, 1])
        tied = allx.best_iou(np.ones((2, 2)), gt)
        self.assertAlmostEqual(tied['best_iou'], 2/3)
        self.assertIsNone(allx.best_iou(np.ones((2, 2)), {'mask': np.zeros((2, 2))}))

    def test_resnet_lrp_conversion_preserves_logits_and_hooks(self):
        model = resnet50(weights=None).eval()
        converted = allx.lrp_model(model)
        image = torch.rand(1, 3, 32, 32, requires_grad=True)
        with torch.no_grad():
            torch.testing.assert_close(model(image), converted(image), atol=1e-5, rtol=1e-4)
        from captum.attr import LRP
        result = LRP(converted).attribute(image, target=0)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(any(module._forward_hooks for module in model.modules()))

    def test_gradient_methods_and_crop_alignment(self):
        # An inexpensive differentiable classifier with the same preprocessing contract.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.ReLU())
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
                self.head = torch.nn.Linear(2, 2)
            def forward(self, x):
                return self.head(self.pool(self.layer4(x)).flatten(1))
        adapter = SimpleNamespace(model=Model().eval(), device=torch.device('cpu'),
                                  preprocess=ResNet50_Weights.DEFAULT.transforms(), name='resnet50')
        image = np.full((32, 64, 3), 150, dtype=np.uint8)
        engine = allx.GradientMethods(adapter, image, [np.zeros_like(image)], self.args())
        for method in ['LRP', 'GradCAM', 'IDG', 'GradExp', 'GradShap']:
            result = engine.compute(method, 0)
            self.assertEqual(result.shape, (1, 32, 64), method)
            self.assertTrue(np.isfinite(result).all(), method)
        restored = engine.restore(torch.ones(1, 3, 224, 224))
        self.assertEqual(float(restored[0, 16, 0]), 0.)
        self.assertGreater(float(restored[0, 16, 32]), 0.)
        adapter.name = 'vit_b_16'
        with self.assertRaisesRegex(allx.UnsupportedMethod, 'transformer-specific'):
            engine.compute('LRP', 0)

    def test_asymmetric_gt_contour_uses_image_coordinates(self):
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from unittest.mock import patch
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        mask = np.zeros((8, 12), dtype=bool)
        mask[:2, 1:4] = True  # Top left; a vertical flip would place it at bottom.
        contours = []
        original = Axes.contour
        def capture(ax, *args, **kwargs):
            result = original(ax, *args, **kwargs)
            contours.append(result)
            return result
        with tempfile.TemporaryDirectory() as temporary, patch.object(Axes, 'contour', capture):
            result = allx.comparison_plot(
                SimpleNamespace(plt=plt), image, {'BPT-8': mask[None].astype(float)},
                {'explained_class': 'object'}, Path(temporary) / 'asymmetric.png',
                {'mask': mask, 'status': 'matched'}, {'BPT-8': {'status': 'complete'}})
        vertices = np.concatenate(contours[0].allsegs[0])
        np.testing.assert_allclose(vertices.min(axis=0), [.5, -.5])
        np.testing.assert_allclose(vertices.max(axis=0), [3.5, 1.5])
        self.assertEqual(result['BPT-8']['best_iou'], 1.)

    def test_comparison_records_iou_and_missing_method(self):
        import matplotlib.pyplot as plt
        runtime = SimpleNamespace(plt=plt)
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, 2:4] = True
        status = {'BPT-8': {'status': 'complete'}, 'LRP': {'status': 'unsupported', 'reason': 'test backend'}}
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'comparison.png'
            result = allx.comparison_plot(runtime, image, {'BPT-8': mask[None].astype(float)},
                                         {'explained_class': 'object'}, target,
                                         {'mask': mask, 'status': 'matched'}, status)
            self.assertEqual(result['BPT-8']['best_iou'], 1.)
            self.assertNotIn('LRP', result)
            self.assertTrue(target.is_file())


if __name__ == '__main__':
    unittest.main()
