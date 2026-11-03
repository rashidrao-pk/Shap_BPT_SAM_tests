"""Requested AUC point counts versus actual distinct-mask evaluations."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'),
               str(Path(__file__).resolve().parents[1] / 'src/scripts')]
from experiment_common import load_runtime, create_parser


class AucPointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_runtime()

    @staticmethod
    def predict(masks):
        return np.asarray(masks).mean(axis=(1, 2))[:, None]

    def test_default_and_override_for_all_entrypoints(self):
        for kind in ['OC_ResNet','OC_ViT','COD_ViT','OD_DETR','OC_CelebA']:
            self.assertEqual(create_parser(kind).parse_args([]).auc_num_samples, 200)
            self.assertEqual(create_parser(kind).parse_args(['--auc-num-samples','350']).auc_num_samples, 350)
        import OD_yolo
        with patch.object(sys, 'argv', ['OD_yolo.py', '--auc-num-samples', '350']):
            self.assertEqual(OD_yolo.parse_args().auc_num_samples, 350)

    def test_200_points_and_tied_map_actual_count(self):
        for direction in ['ins', 'del']:
            dense = self.runtime.utx.saliency_to_auc(self.predict, np.arange(400, dtype=float).reshape(20,20),
                                                    1., 0., 0, method=direction, num_samples=200)
            self.assertEqual(dense['requested_points'], 200)
            self.assertEqual(dense['evaluated_points'], 200)
            self.assertEqual(len(dense['xs']), 200)
            tied = self.runtime.utx.saliency_to_auc(self.predict, np.zeros((20,20)),
                                                   1., 0., 0, method=direction, num_samples=200)
            self.assertEqual(tied['requested_points'], 200)
            self.assertEqual(tied['evaluated_points'], 2)
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            self.runtime.utx.saliency_to_auc(self.predict, np.zeros((2,2)), 1., 0., 0, num_samples=1)

    def test_plot_reports_requested_and_actual_points(self):
        results = []
        for label, heatmap in [('dense', np.arange(400,dtype=float).reshape(20,20)), ('tied',np.zeros((20,20)))]:
            results.append({'label': label, **{f'auc_{direction}': self.runtime.utx.saliency_to_auc(
                self.predict, heatmap, 1., 0., 0, method=direction, num_samples=200) for direction in ['ins','del']}})
        with patch.object(self.runtime.plt,'show'):
            fig, axes = self.runtime.utx.plot_auc_results(results, '.', destroy_figs=False)
        try:
            self.assertIn('requested 200 points', axes[0].get_title())
            labels = [line.get_label() for line in axes[0].lines]
            self.assertTrue(any('(n=200)' in label for label in labels))
            self.assertTrue(any('(n=2)' in label for label in labels))
        finally:
            self.runtime.plt.close(fig)

    def test_yolo_helper_can_disable_hidden_extra_sweeps(self):
        utx = self.runtime.utx
        with patch.object(utx, 'saliency_to_auc', wraps=utx.saliency_to_auc) as auc, \
             patch.object(utx, 'compute_extra_faithfulness_metrics', return_value={}):
            results = utx.compute_auc_results({'test':np.zeros((1,2,2))}, self.predict, 1., 0., 0,
                                              num_samples=200, extra_samples=())
        self.assertEqual(auc.call_count, 2)
        self.assertEqual(results[0]['auc_requested_points'], 200)
        self.assertNotIn('auc_ins_500', results[0])


if __name__ == '__main__':
    unittest.main()
