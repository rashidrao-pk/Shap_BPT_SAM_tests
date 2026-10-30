"""Progress from completion markers, independent of optional plots and AUC."""
from pathlib import Path
import sys
import json
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
import check_experiment_progress as progress


class ExplanationProgressTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_shared_marker_works_without_auc_and_heatmaps(self):
        self.write('run_info.json', {'runner': 'OC_ViT', 'signature': 'current', 'image_ids': ['a','b','c'],
                                    'methods': ['BPT-500', 'LRP']})
        self.write('a/a_predictions.json', {})
        self.write('a/a_method_status.json', {'BPT-500': {'status': 'complete'}, 'LRP': {'status': 'unsupported'}})
        self.write('a/a_complete.json', {'signature': 'current', 'total_time_sec': 12,
                   'artifacts': ['a_predictions.json', 'a_method_status.json']})
        self.write('b/b_predictions.json', {})
        self.write('failures.json', [{'image_id': 'c'}])
        summary, pending = progress.summarize_experiment(self.root, {'a','b','c'}, ['BPT-500','LRP'], False)
        self.assertEqual(summary['completed_images'], 1)
        self.assertEqual(summary['pending_images'], 2)
        self.assertEqual(summary['partial_images'], 1)
        self.assertEqual(summary['failed_images'], 1)
        self.assertEqual(summary['estimated_time_required_sec'], 24)
        self.assertEqual(len(pending), 2)
        (self.root / 'a/a_predictions.json').unlink()
        summary, _ = progress.summarize_experiment(self.root, {'a','b','c'}, ['BPT-500','LRP'], False)
        self.assertEqual(summary['completed_images'], 0)

    def test_celeba_unpadded_and_yolo_all_markers(self):
        self.write('0/0_predictions.json', {})
        self.write('0/0_complete.json', {'artifacts': ['0_predictions.json']})
        self.assertTrue(progress.output_status_for_image(self.root, '000000000000', [], False)[0])
        self.write('139/139_predictions.json', {})
        self.write('139/139_all_complete.json', {'artifacts': ['139_predictions.json']})
        self.assertTrue(progress.output_status_for_image(self.root, '000000000139', [], False)[0])

    def test_old_signature_and_corrupt_marker_are_not_complete(self):
        self.write('run_info.json', {'signature': 'new'})
        self.write('1/1_predictions.json', {})
        marker = self.write('1/1_complete.json', {'signature': 'old','artifacts':['1_predictions.json']})
        self.assertFalse(progress.output_status_for_image(self.root, '000000000001', [], False)[0])
        marker.write_text('{')
        self.assertFalse(progress.output_status_for_image(self.root, '000000000001', [], False)[0])

    def test_csv_eta_excludes_previous_partition_generation(self):
        csv = self.root / 'auc_results_all_images_current_run.csv'
        csv.write_text('image_id,method,time_exp,time_eval\n1,BPT,2,3\n')
        (self.root/'partition_summary.csv').write_text('image_id,total_time_sec\n1,100\n')
        estimate = progress.estimate_time_progress(self.root, csv, ['000000000001'], 2)
        self.assertEqual(estimate['avg_time_per_completed_image_sec'], 5.)
        self.assertEqual(estimate['estimated_time_required_sec'], 10.)

    def test_partition_summary_selected_ids_precede_global_dataset(self):
        (self.root/'partition_summary.csv').write_text('image_id\n139\n')
        self.assertEqual(progress.expected_image_ids_for_experiment(self.root, {'unrelated'}), {'000000000139'})
        (self.root/'ILSVRC2012_val_00000001_compact.npy').touch()
        self.assertIn('ILSVRC2012_val_00000001', progress.expected_image_ids_from_masks(self.root))


if __name__ == '__main__':
    unittest.main()
