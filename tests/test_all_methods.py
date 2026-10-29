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
            '--explain_all', '--bpt-budgets', '8', '--lime-budgets', '8', '--aa-budget', '8',
            '--gradient-samples', '2', '--idg-steps', '4', '--attribution-batch-size', '2'])

    def test_requested_default_matrix_and_budget_routing(self):
        from unittest.mock import Mock, patch
        args = common.create_parser('OC_ResNet').parse_args(['--explain_all'])
        expected = ['BPT-500', 'BPT-1000', 'BPT+SAM_c-500', 'BPT+SAM_c-1000',
                    'SAM_c+KernelSHAP-500', 'SAM_c+KernelSHAP-1000', 'LIME-1000',
                    'Shap-AA-1000', 'LRP', 'GradCAM', 'IDG', 'GradExp', 'GradShap']
        self.assertEqual(allx.method_names(args), expected)
        self.assertEqual(allx.lime_segment_count(args, 'MS_COCO'), 250)
        self.assertEqual(allx.lime_segment_count(args, 'ImageNet'), 100)
        self.assertEqual(allx.lime_segment_count(args, 'CAMO'), 100)
        shape = (3, 4)
        image = np.zeros((*shape, 3), dtype=np.uint8)
        labels = np.indices(shape)[1] // 2
        original_tree, sam_tree = object(), object()
        explainer = SimpleNamespace(explain_instance=Mock(return_value=np.ones((1, *shape))))
        kernel = Mock(return_value=np.ones((1, *shape)))
        build = Mock(return_value=sam_tree)
        runtime = SimpleNamespace(shap_bpt=SimpleNamespace(build_bpt_from_image=build),
                                  utx=SimpleNamespace(kernel_shap_partition_regions=kernel))
        with patch.object(allx, 'lime_map', return_value=(np.ones((1, *shape)), {'requested_regions': 250})):
            results = list(allx.compute_extra(runtime, args, SimpleNamespace(), image, [image],
                                              explainer, original_tree, 0, labels=labels,
                                              predict=lambda _: None, dataset_name='MS_COCO'))
        self.assertEqual([call.kwargs['n_samples'] for call in kernel.call_args_list], [500, 1000])
        self.assertEqual([call.args[0] for call in explainer.explain_instance.call_args_list],
                         [500, 1000, 500, 1000, 1000])
        self.assertEqual([call.kwargs['bpt'] for call in explainer.explain_instance.call_args_list],
                         [original_tree, original_tree, sam_tree, sam_tree, None])
        build.assert_called_once()
        self.assertEqual(results[7][0], 'Shap-AA-1000')
        self.assertEqual(results[7][2]['budget'], 1000)
        args.lime_segments = 37
        self.assertEqual(allx.lime_segment_count(args, 'MS_COCO'), 37)

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

    def test_notebook_style_and_asymmetric_gt_boundaries(self):
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from unittest.mock import patch
        from skimage.segmentation import find_boundaries
        from attribution_visuals import notebook_style
        import shap_bpt
        values = np.arange(100, dtype=float).reshape(10, 10)
        style = notebook_style(values)
        self.assertEqual(style['vmax'], np.quantile(np.abs(values), .99))
        self.assertEqual(style['vmin'], -style['vmax'])
        self.assertIs(style['cmap'], shap_bpt.shapley_values_colormap)
        # Including zeros differs from the old nonzero-only percentile.
        sparse = np.zeros((10, 10))
        sparse[0, 0] = 100
        self.assertAlmostEqual(notebook_style(sparse)['vmax'], 1.)
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        mask = np.zeros((8, 12), dtype=bool)
        mask[:2, 1:4] = True
        images = []
        def capture(fig, *args, **kwargs):
            self.assertEqual(len(fig.axes), 4)  # Two rows: Input and one method, no GT column.
            self.assertNotIn('GT', [ax.get_title() for ax in fig.axes])
            for ax in fig.axes:
                self.assertTrue(ax.axison)
                self.assertEqual(len(ax.get_xticks()), 0)
                self.assertEqual(len(ax.get_yticks()), 0)
                self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
            images.extend(fig.axes[1].images)
        with patch.object(Figure, 'savefig', capture):
            result = allx.comparison_plot(
                SimpleNamespace(plt=plt), image, {'BPT-8': mask[None].astype(float)},
                {'explained_class': 'object'}, 'unused.png',
                {'mask': mask, 'status': 'matched'}, {'BPT-8': {'status': 'complete'}})
        overlay = np.asarray(images[1].get_array())
        np.testing.assert_array_equal(overlay[..., 3] > 0, find_boundaries(mask, mode='thick'))
        self.assertFalse(overlay[4:, :, 3].any())
        self.assertTrue((overlay[overlay[..., 3] > 0, :3] == 0).all())
        self.assertEqual(images[1].origin, 'upper')
        self.assertEqual(result['BPT-8']['best_iou'], 1.)

    def test_high_resolution_gt_has_visible_vector_outline(self):
        import matplotlib.pyplot as plt
        from attribution_visuals import draw_gt_boundaries
        from matplotlib.axes import Axes
        from unittest.mock import patch
        mask = np.zeros((1024, 1024), dtype=bool)
        mask[100:300, 200:400] = True
        fig, ax = plt.subplots(figsize=(2.1, 2.1))
        captured = []
        original = Axes.contour
        def capture(ax, *args, **kwargs):
            result = original(ax, *args, **kwargs)
            captured.append(result)
            return result
        try:
            with patch.object(Axes, 'contour', capture):
                draw_gt_boundaries(ax, mask)
            contour = captured[0]
            vertices = np.concatenate(contour.allsegs[0])
            np.testing.assert_allclose(vertices.min(axis=0), [199.5, 99.5])
            np.testing.assert_allclose(vertices.max(axis=0), [399.5, 299.5])
            self.assertGreaterEqual(float(contour.get_linewidths()[0]), 1.)
            self.assertEqual(contour.get_zorder(), 5)
            self.assertTrue(contour.get_path_effects())
            self.assertGreater(ax.get_ylim()[0], ax.get_ylim()[1])
        finally:
            plt.close(fig)

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
