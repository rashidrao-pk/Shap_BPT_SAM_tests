import importlib.util
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1]/'src/hpc/budget_progress.py'
spec = importlib.util.spec_from_file_location('budget_progress_test', path)
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


def test_partial_images_are_visible_and_incomplete_cache_not_counted(tmp_path):
    folder = tmp_path/'pilot'; folder.mkdir()
    study = {'signature': 'fixed', 'state': 'running', 'methods': ['BPT'], 'budgets': [100],
             'reference_budget': 4000, 'repeats': 2,
             'splits': {'calibration': ['image1'], 'validation': ['image2']}}
    (folder/'study.json').write_text(json.dumps(study))
    directory = folder/'image1/BPT'; directory.mkdir(parents=True)
    for name in ['B100_s0', 'B100_s1', 'reference_s0', 'reference_s1']:
        (directory/(name+'.json')).write_text(json.dumps({'signature': 'fixed',
            'explanation_sec': 10, 'evaluation_sec': 2}))
    (directory/'B100_s0.npz').touch()
    task = {'id': 'budget_fixture', 'output': str(folder)}
    row = progress.snapshot(task, tmp_path)
    assert row['saved'] == 1 and row['total'] == 8
    assert row['images_cached'] == 0 and row['eta_evaluation_seconds'] is None
    for name in ['B100_s1', 'reference_s0', 'reference_s1']:
        (directory/(name+'.npz')).touch()
    row = progress.snapshot(task, tmp_path)
    assert row['saved'] == 4 and row['images_cached'] == 1
    assert row['current_image'] == 'image2'
    assert row['eta_evaluation_seconds'] == 48
    # A marker from incompatible settings must never count as completed work.
    (directory/'B100_s0.json').write_text(json.dumps({'signature': 'different'}))
    assert progress.snapshot(task, tmp_path)['saved'] == 3
