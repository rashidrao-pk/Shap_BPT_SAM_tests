"""xai_timing: nested steps, own times, GPU/CPU/IO split."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'))
import xai_timing  # noqa: E402


def test_nested_steps_own_time_and_kinds(tmp_path):
    timer = xai_timing.start('139')
    with timer.step('method:BPT', 'mixed'):
        time.sleep(.02)  # explainer bookkeeping (CPU)
        with xai_timing.current().step('yolo.predict', 'mixed'):
            xai_timing.current().add('inference (GPU)', .05, 'gpu')
            xai_timing.current().add('preprocess (CPU)', .01, 'cpu')
            xai_timing.current().count('model_images', 10)
    timer.begin('save', 'io')
    timer.end()
    summary = timer.write(tmp_path / '139_timing.json', extra={'runner': 'test'})
    xai_timing.stop()
    steps = {step['step']: step for step in summary['steps']}
    assert steps['method:BPT/yolo.predict/inference (GPU)']['seconds'] == .05
    assert steps['method:BPT']['self_seconds'] >= .015  # its own time, children excluded
    assert summary['by_kind_seconds']['gpu'] == .05 and summary['counters']['model_images'] == 10
    assert json.loads((tmp_path / '139_timing.json').read_text())['runner'] == 'test'
    assert xai_timing.current().step('x') is not None  # no-op when no timer is open


def test_open_steps_are_closed_by_summary():
    timer = xai_timing.start()
    timer.begin('auc_metrics', 'mixed')
    assert timer.summary()['steps'][0]['step'] == 'auc_metrics'
    xai_timing.stop()


def test_report_folds_methods_and_splits_kinds(tmp_path):
    import timing_report
    for image_id in ['1', '2']:
        timer = xai_timing.start(image_id)
        for method in ['BPT-1000', 'EAC-1000']:
            with timer.step(f'method:{method}', 'mixed'):
                xai_timing.current().add('inference (GPU)', .1, 'gpu')
        (tmp_path / image_id).mkdir()
        timer.write(tmp_path / image_id / f'{image_id}_timing.json', extra={'budget': 1000})
        xai_timing.stop()
    timing_report.main([str(tmp_path)])
    text = (tmp_path / 'timing_report.md').read_text()
    assert '| */inference (GPU) | gpu | 0.20 |' in text  # both methods added up, per image
    assert '| BPT-1000 |' in text and (tmp_path / 'timing_report.csv').is_file()
