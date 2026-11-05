#!/usr/bin/env python3
"""Paired image-cluster bootstrap analysis of a frozen full-background budget study."""
import argparse
import json
from pathlib import Path
import numpy as np
from budget_selection import load_study, select_budget


def simultaneous_bounds(blocks, resamples=20000, alpha=.05, seed=0):
    """Basic bootstrap simultaneous upper bounds, with losses already margin-normalized.

    One block per dataset, same image rows for every method/model/partition/budget.
    Repeated seeds are averaged within each image before constructing the blocks.
    Coverage is a bootstrap approximation, not an exact finite-sample guarantee.
    """
    rng = np.random.default_rng(seed)
    maximum = np.full(resamples, -np.inf)
    for block in blocks:
        block = np.asarray(block, dtype=float)
        if block.ndim != 2 or len(block) < 2 or not np.isfinite(block).all():
            raise ValueError('Need finite image-level paired losses for at least two images')
        mean = block.mean(axis=0)
        for start in range(0, resamples, 250):
            count = min(250, resamples-start)
            indices = rng.integers(len(block), size=(count, len(block)))
            # max(mean - bootstrap mean) gives simultaneous BASIC upper bounds.
            draws = block[indices].mean(axis=1)
            maximum[start:start+count] = np.maximum(maximum[start:start+count],
                                                   (mean-draws).max(axis=1))
    correction = max(0., float(np.quantile(maximum, 1-alpha)))
    return [np.asarray(block).mean(axis=0)+correction for block in blocks], correction


def analyze(studies, protocol):
    # Reuse the existing complete-design validation; its strict empirical decision is separate.
    empirical = select_budget(studies)
    if any(study['background'] != 'full' for study in studies):
        raise ValueError('This study requires full replacement in every scope')
    margins = protocol['margins']
    budgets = protocol['budgets']
    output = {}
    for split in ['calibration', 'validation']:
        datasets = sorted({study['dataset'] for study in studies})
        blocks, descriptors, stable = [], [], True
        for dataset in datasets:
            group = [(i, study) for i, study in enumerate(studies) if study['dataset'] == dataset]
            ids = sorted(group[0][1]['splits'][split])
            if any(sorted(study['splits'][split]) != ids for _, study in group):
                raise ValueError('Each dataset must share the same split across all scopes')
            columns, names = [], []
            for index, study in group:
                for method in study['methods']:
                    for budget in budgets:
                        for metric, margin in margins.items():
                            values = []
                            for image in ids:
                                rows = [row for row in study['rows'] if row['image_id'] == image and
                                        row['method'] == method and row['budget'] == budget]
                                stable &= all(row['reference_stable'] and row['informative_reference'] for row in rows)
                                losses = [(1-row['spearman'] if metric == 'spearman_loss' and row['spearman'] is not None
                                           else 1-row['top_jaccard'] if metric == 'top_jaccard_loss'
                                           else row.get(metric)) for row in rows]
                                if any(value is None or not np.isfinite(value) for value in losses):
                                    raise ValueError('Noninformative comparisons cannot support non-inferiority')
                                values.append(float(np.mean(losses))/margin)
                            columns.append(values)
                            names.append({'study': index, 'runner': study['runner'], 'dataset': dataset,
                                'scope': study['path'], 'method': method, 'budget': budget,
                                'metric': metric, 'margin': margin, 'images': len(ids)})
            blocks.append(np.asarray(columns).T)
            descriptors.append(names)
        bounds, correction = simultaneous_bounds(blocks, protocol['bootstrap_resamples'],
                                                  protocol['alpha'], protocol['seed'])
        rows = []
        for block, upper, names in zip(blocks, bounds, descriptors):
            for mean, bound, name in zip(block.mean(axis=0), upper, names):
                rows.append({**name, 'mean_loss': float(mean*name['margin']),
                             'upper_bound': float(bound*name['margin']), 'pass': bool(bound < 1)})
        output[split] = {'reference_stable': bool(stable), 'normalized_correction': correction, 'bounds': rows}
    candidate = next((budget for budget in budgets if output['calibration']['reference_stable'] and all(
        row['pass'] for row in output['calibration']['bounds'] if row['budget'] >= budget)), None)
    validated = candidate is not None and output['validation']['reference_stable'] and all(
        row['pass'] for row in output['validation']['bounds'] if row['budget'] >= candidate)
    return {'status': 'validated' if validated else 'validation_failed' if candidate is not None else 'no_acceptable_budget',
        'calibration_candidate': candidate, 'selected_budget': candidate if validated else None,
        'protocol': protocol, 'splits': output, 'strict_empirical_status': empirical['status'],
        'claim': 'Approximate simultaneous bootstrap evidence of mean reference agreement on tested scopes, '
                 'not correctness, exact Shapley values, or a universal minimum. Images are resampled, not repeats.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_root', type=Path)
    parser.add_argument('--manifest', default='manifest.json', help='Frozen task manifest to analyze')
    parser.add_argument('--report-dir', default='report', help='Separate report folder for this scope')
    args = parser.parse_args()
    home = args.study_root.resolve()
    manifest = json.loads((home/args.manifest).read_text())
    studies, pending = [], []
    for task in manifest['tasks']:
        if task['type'] != 'budget':
            raise ValueError('Budget-only manifest required')
        folder = home / Path(task['output']).relative_to(manifest['study_root'])
        if not (folder/'study.json').is_file() or json.loads((folder/'study.json').read_text()).get('state') != 'complete':
            pending.append(task['id'])
        else:
            studies.append(load_study(folder))
    report = home/args.report_dir; report.mkdir(parents=True, exist_ok=True)
    if pending:
        result = {'status': 'incomplete', 'completed': len(studies), 'pending': pending, 'selected_budget': None}
        conclusion = f"Budget ablation is incomplete: {len(studies)}/{len(manifest['tasks'])} pilots finished. No recommendation yet."
    else:
        result = analyze(studies, manifest['budget_protocol'])
        conclusion = (f"Recommended common XAI budget: {result['selected_budget']} with full background replacement."
                      if result['selected_budget'] is not None else
                      'No tested common XAI budget passed the predeclared non-inferiority and reference-stability checks. '
                      'Do not assume 1000, 2000, or the 4000 reference is sufficient.')
    (report/'noninferiority.json').write_text(json.dumps(result, indent=2)+'\n')
    (report/'conclusion.md').write_text('# Full-background XAI budget ablation\n\n'+conclusion+
        '\n\nBudgets: 100, 200, 500, 1000, 2000; approximate reference: 4000. '
        'AUC resolution is fixed at 200, not swept.\n\n'
        'The analysis tests mean agreement with the reference, not attribution correctness. '
        'A recommendation applies only to the tested datasets, models, partitions and target settings.\n')
    print(conclusion)


if __name__ == '__main__':
    main()
