"""Run with: python -m pytest tests/test_busi_tree_attribution.py"""
import importlib.util
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'src/scripts/run_busi_shapbpt.py'
spec = importlib.util.spec_from_file_location('busi_pipeline', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_hierarchical_permutation():
    nodes = {0: {'children': []}, 1: {'children': []}, 2: {'children': []},
             3: {'children': [0, 1]}, 4: {'children': [3, 2]}}
    rng = np.random.default_rng(0)
    for _ in range(30):
        order = mod.random_hierarchical_order(nodes, 4, rng)
        assert sorted(order) == [0, 1, 2]
        assert abs(order.index(0) - order.index(1)) == 1


def test_localization():
    heat = np.array([[3., 1.], [0., -1.]])
    gt = np.array([[1, 0], [0, 0]], dtype=bool)
    m = mod.localization(heat, gt)
    assert m['pointing'] is True
    assert m['top_area_iou'] == 1.0


def test_discover_cases_excludes_masks(tmp_path):
    sub = tmp_path / 'benign'
    sub.mkdir()
    (sub / 'benign (1).png').touch()
    (sub / 'benign (1)_mask.png').touch()
    (sub / 'benign (1)_mask_1.png').touch()
    cases = mod.discover_cases(tmp_path)
    assert len(cases) == 1 and len(cases[0]['masks']) == 2


def test_download_does_not_import_inference_dependencies(tmp_path):
    """Execute the real CLI with network clients stubbed and scientific imports blocked."""
    import subprocess
    import sys
    harness = '''
import importlib.abc, runpy, sys, types
from pathlib import Path
class BlockScientific(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'PIL', 'torch', 'transformers', 'matplotlib', 'scipy', 'kagglehub', 'IPython', 'psutil'}:
            raise AssertionError('Unexpected inference dependency: ' + fullname)
sys.meta_path.insert(0, BlockScientific())
import urllib.request, zipfile
def fake_retrieve(url, destination):
    with zipfile.ZipFile(destination, 'w') as archive:
        archive.writestr('Dataset_BUSI_with_GT/benign/benign (1).png', b'fixture')
    return destination, None
urllib.request.urlretrieve = fake_retrieve
import os
def fake_snapshot(*args, **kwargs):
    assert os.environ.get('HF_HUB_DISABLE_XET') == '1'
sys.modules['huggingface_hub'] = types.SimpleNamespace(snapshot_download=fake_snapshot)
script, folder = sys.argv[1:]
sys.argv = [script, 'download', '--dataset', folder + '/data', '--model-dir', folder + '/model']
runpy.run_path(script, run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', harness, str(SCRIPT), str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '[cases] 1' in result.stdout


def test_dataset_rejects_html_response(tmp_path, monkeypatch):
    import urllib.request
    import pytest
    def retrieve(url, destination):
        Path(destination).write_text('<html>Sign in</html>')
    monkeypatch.setattr(urllib.request, 'urlretrieve', retrieve)
    with pytest.raises(RuntimeError, match='not a ZIP'):
        mod.download_dataset(tmp_path)
    assert not (tmp_path / 'BUSI_kaggle.zip').exists()


def test_existing_dataset_skips_network(tmp_path, monkeypatch):
    import urllib.request
    folder = tmp_path / 'benign'
    folder.mkdir()
    (folder / 'benign (1).png').touch()
    monkeypatch.setattr(urllib.request, 'urlretrieve', lambda *a: (_ for _ in ()).throw(AssertionError('network')))
    mod.download_dataset(tmp_path)


def test_resume_after_interruption_skips_completed_images(tmp_path, monkeypatch):
    import json
    import sys
    import pytest
    data = tmp_path / 'data/benign'
    data.mkdir(parents=True)
    for name in ['benign (1).png', 'benign (2).png']:
        (data / name).touch()
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.safetensors').touch()
    checkpoint = tmp_path / 'sam.pt'
    checkpoint.touch()
    results = tmp_path / 'results'
    argv = [str(SCRIPT), 'run', '--device', 'cpu', '--dataset', str(data.parent),
            '--model-dir', str(model), '--sam', 'sam1', '--sam1-checkpoint', str(checkpoint),
            '--results', str(results), '--limit', '2']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(mod, 'load_classifier', lambda *a: (None, None, None))
    calls = []
    def process(case, args, *unused):
        calls.append(Path(case['image']).name)
        if len(calls) == 2:
            raise KeyboardInterrupt('simulate wall-time interruption')
        folder = mod.case_folder(case, args)
        for artifact in mod.case_artifacts(args):
            path = folder / artifact
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}' if artifact == 'metrics.json' else 'fixture')
        return {}
    monkeypatch.setattr(mod, 'process_one', process)
    with pytest.raises(KeyboardInterrupt):
        mod.main()
    assert len(list(results.rglob('complete.json'))) == 1
    mod.main()
    assert calls == ['benign (1).png', 'benign (2).png', 'benign (2).png']
    assert len(list(results.rglob('complete.json'))) == 2
    monkeypatch.setattr(mod, 'load_classifier', lambda *a: (_ for _ in ()).throw(AssertionError('model loaded')))
    mod.main()  # Entirely complete runs require no model load.
    assert len(json.loads((results / 'summary.json').read_text())) == 2
    monkeypatch.setattr(sys, 'argv', [*argv, '--budget', '1000'])
    with pytest.raises(ValueError, match='settings/inputs changed'):
        mod.main()


def test_explain_all_dispatches_to_shared_runner_for_selected_sam(tmp_path, monkeypatch):
    import json
    import sys
    import yaml
    sys.path.insert(0, str(SCRIPT.parents[0] / 'explanations'))
    import experiment_common as common
    base = tmp_path / 'partitions'
    sam1 = base / 'BUSI_sam1_vit_b_62_compact-sam_50'
    sam2 = base / 'BUSI_sam2_tiny_62_compact-sam_50'
    sam1.mkdir(parents=True)
    sam2.mkdir()
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data': {'name': 'BUSI', 'mask_dir_final': sam2.name},
                                    'output': {'partitions': str(base)}}))
    captured = []
    monkeypatch.setattr(common, 'run', lambda kind, args: captured.append((kind, args)))
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), 'run', '--config', str(config),
        '--sam', 'sam1', '--model-dir', str(tmp_path / 'model'), '--results', str(tmp_path / 'results'),
        '--limit', '1', '--budget', '1000', '--explain_all', '--no-heatmaps', '--show-gt',
        '--batch-size', '8', '--auc-num-samples', '200'])
    mod.main()
    assert len(captured) == 1
    kind, args = captured[0]
    assert kind == 'OC_BUSI' and args.explain_all and args.max_evals == 1000
    assert args.auc_num_samples == 200 and args.model_batch_size == 8
    assert args.bg_type == 'full'
    assert args.show_gt and not args.save_heatmaps and args.resume
    written = yaml.safe_load(Path(args.config).read_text())
    assert written['data']['mask_dir_final'] == str(sam1)
    assert yaml.safe_load(config.read_text())['data']['mask_dir_final'] == sam2.name


def test_legacy_coalitions_average_each_background(monkeypatch):
    from PIL import Image
    image = Image.fromarray(np.full((2, 2, 3), 100, dtype=np.uint8))
    labels = np.array([[0, 0], [1, 1]])
    baselines = [Image.fromarray(np.full((2, 2, 3), value, dtype=np.uint8)) for value in [0, 20, 40, 60, 80]]
    captured = []
    def score(images, *args):
        captured.extend(images)
        values = np.array([np.asarray(im).mean() for im in images])
        return values, values/100
    monkeypatch.setattr(mod, 'score_images', score)
    scores, probabilities = mod.score_coalitions(image, labels, [set(), {0}, {0, 1}],
                                                 baselines, None, None, 'cpu', 0, 8)
    np.testing.assert_allclose(scores, [40, 70, 100])
    np.testing.assert_allclose(probabilities, [.4, .7, 1.])
    assert len(captured) == 15
