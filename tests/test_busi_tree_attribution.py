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
        if fullname.split('.')[0] in {'numpy', 'PIL', 'torch', 'transformers', 'matplotlib', 'scipy'}:
            raise AssertionError('Unexpected inference dependency: ' + fullname)
sys.meta_path.insert(0, BlockScientific())
def dataset_download(dataset, output_dir):
    folder = Path(output_dir) / 'benign'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'benign (1).png').touch()
    return output_dir
sys.modules['kagglehub'] = types.SimpleNamespace(dataset_download=dataset_download)
sys.modules['huggingface_hub'] = types.SimpleNamespace(snapshot_download=lambda *a, **k: None)
script, folder = sys.argv[1:]
sys.argv = [script, 'download', '--dataset', folder + '/data', '--model-dir', folder + '/model']
runpy.run_path(script, run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', harness, str(SCRIPT), str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '[cases] 1' in result.stdout
