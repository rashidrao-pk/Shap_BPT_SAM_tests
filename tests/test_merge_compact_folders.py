"""src/hpc/merge_compact_folders.py merges <base>_greedy / _mso_<a> / _split folders into <base>."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'src/hpc/merge_compact_folders.py'
BASE = 'MSCOCO_sam2_large_254_compact-sam_200'


def folder(root, suffix, sam, compact, image_id='000000000139'):
    path = root / f'{BASE}_{suffix}'
    path.mkdir()
    np.save(path / f'{image_id}_sam.npy', sam)
    np.save(path / f'{image_id}_compact.npy', compact)
    (path / f'{image_id}_results.json').write_text(json.dumps({
        'n_sam_masks': 5, 'saved_partitions': {'sam': 'x', 'compact': 'y'},
        'partition_label_counts': {'sam': 4, 'compact': int(compact.max()) + 1},
        'generation_settings': {'compact_strategy': suffix}}))
    return path


def run(root, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), '--root', str(root), *extra],
                          capture_output=True, text=True, check=True).stdout


def test_merge(tmp_path):
    sam = np.arange(12, dtype=np.uint16).reshape(3, 4)
    folder(tmp_path, 'greedy', sam, np.ones((3, 4), np.uint16))
    folder(tmp_path, 'split', sam, np.full((3, 4), 2, np.uint16))
    folder(tmp_path, 'mso_0.5', sam + 1, np.full((3, 4), 3, np.uint16))  # a different SAM run
    busy = folder(tmp_path, 'split_200', sam, sam)
    (busy / '000000000140_compact.npy.tmp').write_bytes(b'')

    report = run(tmp_path)
    assert 'Dry run' in report and not (tmp_path / BASE).exists()

    report = run(tmp_path, '--apply')
    target = tmp_path / BASE
    names = sorted(path.name for path in target.iterdir())
    assert names == ['000000000139_compact_greedy.npy', '000000000139_compact_mso_0.5.npy',
                     '000000000139_compact_split.npy', '000000000139_results.json', '000000000139_sam.npy',
                     '000000000139_sam_mso_0.5.npy']
    assert np.load(target / '000000000139_compact_split.npy').max() == 2
    metadata = json.loads((target / '000000000139_results.json').read_text())
    assert set(metadata['compact_files']) == {'greedy', 'split', 'mso_0.5'}
    assert metadata['compact_files']['mso_0.5']['sam_run_differs'] is True
    assert metadata['compact_files']['split']['sam_run_differs'] is False
    assert {'compact_greedy', 'compact_split', 'compact_mso_0.5', 'sam'} <= set(metadata['saved_partitions'])
    assert 'compact' not in metadata['saved_partitions']
    assert not (tmp_path / f'{BASE}_greedy').exists() and not (tmp_path / f'{BASE}_split').exists()
    assert busy.exists() and 'SKIP' in report  # A folder being written is left alone.

    assert 'No folders' not in run(tmp_path, '--apply')  # Rerun: only the busy folder remains, still skipped.
