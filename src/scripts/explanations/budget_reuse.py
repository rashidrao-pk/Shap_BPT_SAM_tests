"""Verify that reused map evaluations have the same computation and inputs."""
import json
from pathlib import Path


def validate_reuse(study, source_folder):
    source = json.loads((Path(source_folder)/'study.json').read_text())
    keys = ['runner', 'dataset', 'background', 'budgets', 'reference_budget', 'repeats', 'pilot_seed',
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
