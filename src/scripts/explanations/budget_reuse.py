"""Verify that reused map evaluations have the same computation and inputs."""
import json
from pathlib import Path


def validate_reuse(study, source_folder):
    source = json.loads((Path(source_folder)/'study.json').read_text())
    # A map and its own AUC are independent of which OTHER budget is used as reference.
    # Comparisons and reference means must still be recomputed by the new study.
    keys = ['runner', 'dataset', 'background', 'repeats', 'pilot_seed',
            'auc_points', 'compute_auc', 'criteria', 'splits', 'inputs', 'eligible_images', 'versions', 'model_files']
    changed = [key for key in keys if study.get(key) != source.get(key)]
    computation = ['model', 'checkpoint', 'weights', 'class_names', 'attribute', 'attribute_file',
        'target_class', 'class_aggregation', 'threshold', 'detr_shortest_edge', 'eval_batch_size',
        'model_batch_size', 'auc_batch_size', 'kernel_shap_batch_size', 'kernel_shap_samples']
    changed += ['settings.'+key for key in computation
                if study['settings'].get(key) != source.get('settings', {}).get(key)]
    if not set(study['methods']).issubset(source['methods']):
        changed.append('methods not present in source')
    if changed:
        raise ValueError('Cannot reuse incompatible pilot evaluations: '+', '.join(changed))
    return source


def find_reusable_cache(directory, signature, method, budget, seed):
    """Match actual budget/method, never mistake an old reference for a new one.

    Deterministic BPT maps may move between reference/candidate roles across seeds.
    Stochastic methods require an exact seed match.
    """
    directory = Path(directory)
    candidates = []
    for path in sorted(directory.glob('*.json')):
        if not path.with_suffix('.npz').is_file():
            continue
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if row.get('signature') != signature or row.get('method') != method or row.get('budget') != budget:
            continue
        if row.get('seed') == seed:
            return path, row
        if method in {'BPT', 'BPT+SAM_c'}:
            candidates.append((path, row))
    return candidates[0] if candidates else None
