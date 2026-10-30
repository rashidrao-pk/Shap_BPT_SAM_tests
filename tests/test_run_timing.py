"""Wall-clock summary output, persistence, and resumed-run semantics."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from run_timing import duration, report_timing


class RunTimingTest(unittest.TestCase):
    def test_single_image_summary_and_saved_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            rows = [{'image_id': '139', 'total_time_sec': 125.5,
                     'explanation_time_sec': 100., 'evaluation_time_sec': 20.}]
            with redirect_stdout(output):
                summary = report_timing(root/'xai', root/'eval', rows, 3., 130.)
            self.assertIn('Image 139: 2m 5.50s (125.50 seconds)', output.getvalue())
            self.assertIn('Model loading: 3.00s', output.getvalue())
            self.assertEqual(summary['computed_images'], 1)
            self.assertEqual(summary['average_image_sec'], 125.5)
            for folder in ['xai', 'eval']:
                self.assertEqual(json.loads((root/folder/'timing_summary.json').read_text()), summary)

    def test_cached_images_are_not_timed_as_new_work(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            summary = report_timing(directory, directory, [], 0., .1, skipped=2)
            self.assertEqual(summary['computed_images'], 0)
            self.assertEqual(summary['skipped_images'], 2)
            self.assertIsNone(summary['average_image_sec'])
            self.assertIn('No new images completed', output.getvalue())

    def test_hour_duration(self):
        self.assertEqual(duration(3661.25), '1h 1m 1.25s')


if __name__ == '__main__':
    unittest.main()
