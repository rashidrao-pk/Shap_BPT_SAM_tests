"""Predeclared stability criteria and held-out validation for budget pilots.

A high-budget reference is an approximation, not exact Shapley ground truth.
No candidate is selected by looking at held-out failures.
"""
import json
from pathlib import Path

import numpy as np

DEFAULT_CRITERIA = {'min_spearman': .95, 'min_top_jaccard': .90, 'max_relative_l1': .10,
                    'max_auc_difference': .01, 'min_image_pass_fraction': 1.0}


def compare_maps(values, reference, top_fraction=.10):
    from scipy.stats import spearmanr
    a, b = np.asarray(values, dtype=float).ravel(), np.asarray(reference, dtype=float).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Attribution maps must be finite and have matching shapes')
    informative = bool(np.ptp(b) > 1e-12 * max(float(np.max(np.abs(b))), 1.))
    if informative and np.ptp(a) > 0:
        rank = float(spearmanr(a, b).statistic)
    else:
        rank = None
    def support(x):
        absolute = np.abs(x)
        threshold = np.quantile(absolute, 1-top_fraction)
        return (absolute >= threshold) & (absolute > 0)
    left, right = support(a), support(b)
    union = int(np.logical_or(left, right).sum())
    overlap = float(np.logical_and(left, right).sum()/union) if union else 0.
    return {'informative_reference': informative, 'spearman': rank,
            'top_jaccard': overlap,
            'relative_l1': float(np.abs(a-b).sum()/max(float(np.abs(b).sum()), 1e-12))}


def passes(row, criteria, use_auc=True):
    return bool(row.get('informative_reference') and row.get('reference_stable', False)
                and row.get('spearman') is not None and row['spearman'] >= criteria['min_spearman']
                and row['top_jaccard'] >= criteria['min_top_jaccard']
                and row['relative_l1'] <= criteria['max_relative_l1']
                and (not use_auc or (row.get('auc_ins_difference') is not None
                                     and row.get('auc_del_difference') is not None
                                     and row['auc_ins_difference'] <= criteria['max_auc_difference']
                                     and row['auc_del_difference'] <= criteria['max_auc_difference'])))


def select_budget(studies):
    """Choose on calibration only, then accept/reject once on held-out images.

    Every seed must pass for an image/method. Coverage must pass separately in
    every study/method group, at the candidate AND all larger tested budgets.
    An incomplete design never counts as convergence.
    """
    if not studies:
        raise ValueError('At least one study is required')
    first = studies[0]
    design_keys = ['budgets', 'reference_budget', 'criteria', 'repeats', 'compute_auc', 'methods', 'auc_points']
    for study in studies:
        if study.get('state') != 'complete':
            raise ValueError('Study is incomplete; resume it before selecting a budget')
        if any(study.get(key) != first.get(key) for key in design_keys):
            raise ValueError('Studies must have the same budgets, reference, criteria, repeats, methods and AUC settings')
    budgets, criteria = first['budgets'], first['criteria']
    all_rows, design = [], []
    for index, study in enumerate(studies):
        for split in ['calibration', 'validation']:
            if not study['splits'].get(split):
                raise ValueError('Both calibration and held-out validation images are required')
        expected = {(image, method, budget, seed)
                    for ids in study['splits'].values() for image in ids
                    for method in study['methods'] for budget in budgets for seed in range(study['repeats'])}
        rows = study['rows']
        observed = {(row['image_id'], row['method'], row['budget'], row['repeat']) for row in rows}
        if observed != expected or len(rows) != len(expected):
            raise ValueError('Study has missing, duplicate or unexpected comparisons')
        split_for = {image: split for split, ids in study['splits'].items() for image in ids}
        if len(split_for) != sum(len(ids) for ids in study['splits'].values()):
            raise ValueError('Calibration and validation images must be disjoint')
        for row in rows:
            all_rows.append({**row, 'study': index, 'split': split_for[row['image_id']],
                             'pass': passes(row, criteria, first['compute_auc'])})
        design.append({'runner': study['runner'], 'dataset': study['dataset'],
                       'background': study['background'], 'attribute': study.get('attribute'),
                       'model': study.get('settings', {}).get('model'),
                       'splits': study['splits'], 'path': study.get('path')})
    def assess(budget, split):
        groups = []
        for index, study in enumerate(studies):
            for method in study['methods']:
                ids = study['splits'][split]
                outcomes = []
                for image in ids:
                    rows = [row for row in all_rows if row['study'] == index and row['method'] == method
                            and row['image_id'] == image and row['budget'] >= budget]
                    outcomes.append(all(row['pass'] for row in rows))
                fraction = sum(outcomes)/len(outcomes)
                groups.append({'study': index, 'runner': study['runner'], 'method': method,
                               'split': split, 'images': len(ids), 'passed_images': sum(outcomes),
                               'pass_fraction': fraction,
                               'pass': fraction >= criteria['min_image_pass_fraction']})
        return groups
    calibration = {str(budget): assess(budget, 'calibration') for budget in budgets}
    candidate = next((budget for budget in budgets if all(group['pass'] for group in calibration[str(budget)])), None)
    validation = assess(candidate, 'validation') if candidate is not None else []
    accepted = candidate is not None and all(group['pass'] for group in validation)
    validated = accepted and first['compute_auc']
    status = ('validated' if validated else 'stability_only' if accepted else
              'validation_failed' if candidate is not None else 'no_acceptable_budget')
    return {'status': status,
            'selected_budget': candidate if validated else None, 'calibration_candidate': candidate,
            'tested_budgets': budgets, 'reference_budget': first['reference_budget'],
            'criteria': criteria, 'auc_checked': first['compute_auc'],
            'methods': first['methods'], 'auc_points': first['auc_points'],
            'calibration': calibration, 'validation': validation, 'scopes': design,
            'claim': 'Smallest tested budget meeting the declared criteria for these pilot scopes. '
                     'Empirical evidence only; not a universal minimum or exact-SHAP guarantee. '
                     'Held-out validation does not change the calibration candidate.'}


def load_study(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path/'study.json'
    study = json.loads(path.read_text())
    study['rows'] = json.loads((path.parent/'comparisons.json').read_text())
    study['path'] = str(path.parent)
    return study
