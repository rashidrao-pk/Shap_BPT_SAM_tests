#!/usr/bin/env python3
"""Audit and summarize a frozen real-data AUC/budget study without rerunning inference."""
import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from budget_selection import load_study
from budget_ablation import write_reports


DATASETS = {'OC_ResNet': 'ImageNet', 'OC_VGG': 'ImageNet', 'OC_ViT': 'ImageNet',
    'COD_ViT': 'CAMO', 'OC_BUSI': 'BUSI', 'OC_CelebA': 'CelebA-HQ',
    'OC_PartImageNet': 'PartImageNet', 'OC_FunnyBirds': 'FunnyBirds', 'OC_PACO': 'PACO',
    'OC_KvasirSEG': 'Kvasir-SEG', 'OC_ISIC2018': 'ISIC 2018', 'OD_DETR': 'MS-COCO', 'OD_yolo': 'MS-COCO'}


def locate(home, manifest, task):
    remote = manifest.get('study_root')
    if remote is None:
        auc = next((item for item in manifest['tasks'] if item['type'] == 'auc'), None)
        remote = auc['output'].split('/auc/', 1)[0] if auc else task['output'].split('/budget/', 1)[0]
    relative = Path(task['output']).relative_to(remote)
    return home / relative


def audit_curves(directory, frame):
    maximum, count = 0., 0
    for (image, method), group in frame.groupby(['image_id', 'method']):
        path = directory / 'curves' / image / (method + '.json')
        curves = json.loads(path.read_text())
        for _, row in group.iterrows():
            for key, expected in [(f'{row.direction}_{int(row.requested_points)}', row.auc),
                                  (f'{row.direction}_reference', row.reference_auc)]:
                curve = curves[key]
                xs, ys = np.asarray(curve['xs']), np.asarray(curve['y_clipr'])
                if len(xs) != curve['evaluated_points'] or not np.all(np.diff(xs) > 0):
                    raise ValueError(f'Invalid evaluated-point count or curve ordering: {path}/{key}')
                if not np.isfinite(ys).all() or np.min(ys) < -1e-10 or np.max(ys) > 1+1e-10:
                    raise ValueError(f'Invalid normalized curve scores: {path}/{key}')
                integrated = float(np.sum(np.diff(xs) * (ys[:-1] + ys[1:]) / 2))
                error = abs(integrated - expected)
                if error > row.auc_repeat_range + 1e-10:
                    raise ValueError(f'Reported AUC does not match saved curve: {path}/{key}')
                maximum, count = max(maximum, error), count+1
    return count, maximum


def summarize_auc(home, manifest, output):
    scope_rows, measurements = [], []
    checked, discrepancy = 0, 0.
    for task in manifest['tasks']:
        if task['type'] != 'auc':
            continue
        directory = locate(home, manifest, task)
        status = json.loads((directory / 'comparison_status.json').read_text())
        if status.get('state') != 'complete':
            raise ValueError(f'AUC comparison incomplete: {task["id"]}')
        summary = json.loads((directory / 'summary.json').read_text())
        frame = pd.read_csv(directory / 'measurements.csv', dtype={'image_id': str})
        expected = {(case['image_id'], method, direction, n)
                    for case in manifest['auc_cases'] if case['task_id'] == task['id']
                    for method in case['methods'] for direction in ['ins', 'del'] for n in [100, 200]}
        observed = set(frame[['image_id', 'method', 'direction', 'requested_points']].itertuples(index=False, name=None))
        if observed != expected or len(frame) != len(expected):
            raise ValueError(f'Missing or duplicate AUC measurements in {task["id"]}')
        if set(frame.requested_points) != {100, 200} or set(frame.reference_points) != {1000}:
            raise ValueError('Unexpected resolution grid')
        count, delta = audit_curves(directory, frame)
        checked, discrepancy = checked+count, max(discrepancy, delta)
        frame['scope'] = task['id']
        frame['dataset'] = DATASETS[task['runner']]
        frame['runner'] = task['runner']
        frame['background'] = summary['settings']['bg_type']
        measurements.append(frame)
        scope_rows.append({'scope': task['id'], 'dataset': DATASETS[task['runner']],
            'runner': task['runner'], 'background': summary['settings']['bg_type'],
            'images': summary['image_count'], 'methods': summary['method_count'],
            'mean_error_100': summary['mean_error_low'], 'mean_error_200': summary['mean_error_high'],
            'max_error_100': summary['max_error_low'], 'max_error_200': summary['max_error_high'],
            'seconds_100': summary['summed_median_seconds_low'], 'seconds_200': summary['summed_median_seconds_high'],
            'runtime_ratio': summary['summed_median_seconds_high'] / summary['summed_median_seconds_low'],
            'recommendation': summary['recommendation']})
    frame = pd.concat(measurements, ignore_index=True)
    scope_table = pd.DataFrame(scope_rows)
    pairs = frame.pivot(index=['scope', 'image_id', 'method', 'direction'],
                        columns='requested_points', values=['auc', 'absolute_reference_error', 'median_seconds', 'mask_evaluations'])
    low, high = pairs['absolute_reference_error'][100], pairs['absolute_reference_error'][200]
    changes = abs(pairs['auc'][200] - pairs['auc'][100])
    tolerance = manifest['auc_protocol']['tolerance']
    ordering_changes, material_changes, comparable = 0, 0, 0
    for _, group in frame.groupby(['scope', 'image_id', 'direction']):
        values = group.pivot(index='method', columns='requested_points', values='auc')
        for left, right in combinations(values.index, 2):
            comparable += 1
            a = float(values.loc[left, 100] - values.loc[right, 100])
            b = float(values.loc[left, 200] - values.loc[right, 200])
            if np.sign(a) != np.sign(b):
                ordering_changes += 1
                material_changes += max(abs(a), abs(b)) >= tolerance
    times = frame.groupby('requested_points').median_seconds.sum()
    masks = frame.groupby('requested_points').mask_evaluations.sum()
    original_images = {case['image'] for case in manifest['auc_cases']}
    summary = {'state': 'complete', 'scopes': len(scope_rows), 'image_cases': sum(scope_table.images),
        'unique_source_images': len(original_images), 'saved_maps': len(pairs)//2, 'curve_pairs': len(pairs),
        'mean_error_100': float(low.mean()), 'mean_error_200': float(high.mean()),
        'mean_error_reduction_percent': float(100 * (1 - high.mean()/low.mean())),
        'max_error_100': float(low.max()), 'max_error_200': float(high.max()),
        'max_auc_change': float(changes.max()), 'mean_auc_change': float(changes.mean()),
        'seconds_100': float(times[100]), 'seconds_200': float(times[200]),
        'runtime_ratio': float(times[200]/times[100]),
        'masks_100': int(masks[100]), 'masks_200': int(masks[200]),
        'within_tolerance_100': int((low <= tolerance).sum()), 'within_tolerance_200': int((high <= tolerance).sum()),
        'improved': int((high < low-1e-12).sum()), 'worsened': int((high > low+1e-12).sum()),
        'unchanged': int((abs(high-low) <= 1e-12).sum()), 'tolerance': tolerance,
        'max_candidate_repeat_range': float(frame.auc_repeat_range.max()),
        'pairwise_method_order_comparisons': comparable, 'pairwise_order_or_tie_changes': ordering_changes,
        'pairwise_changes_with_gap_at_least_tolerance': int(material_changes),
        'audited_curve_integrations': checked, 'max_auc_integration_discrepancy': discrepancy}
    summary['distinct_saved_curves'] = len(pairs) * 3
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'auc_all_measurements.csv', index=False)
    scope_table.to_csv(output / 'auc_scope_summary.csv', index=False)
    pairs.to_csv(output / 'auc_paired_all.csv')
    (output / 'auc_aggregate.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    labels = [f"{row.runner}\n{row.background}" for row in scope_table.itertuples()]
    positions = np.arange(len(scope_table))
    ax = axes[0, 0]
    ax.bar(positions-.18, scope_table.mean_error_100, .36, label='100 points')
    ax.bar(positions+.18, scope_table.mean_error_200, .36, label='200 points')
    ax.set_xticks(positions, labels, rotation=45, ha='right'); ax.legend()
    ax.set_title('Mean absolute error vs 1000-point reference'); ax.set_ylabel('Normalized AUC error')
    ax = axes[0, 1]
    ax.bar(positions, scope_table.runtime_ratio, color='#507fa6'); ax.axhline(1, color='black', linestyle='--')
    ax.set_xticks(positions, labels, rotation=45, ha='right'); ax.set_ylabel('200-point / 100-point runtime')
    ax.set_title('Runtime cost, sums of median curve times')
    ax = axes[1, 0]
    method_counts = frame.groupby(['method', 'requested_points']).mask_evaluations.mean().unstack()
    method_counts.plot.bar(ax=ax, rot=45)
    ax.set_title('Actual masks evaluated per curve'); ax.set_ylabel('Mean distinct masks')
    ax = axes[1, 1]
    colors = ['#346b96' if name.startswith(('BPT', 'SAM_c', 'LIME', 'Shap-AA')) else '#c06932'
              for name in low.index.get_level_values('method')]
    ax.scatter(low, high, c=colors, alpha=.6, s=25)
    maximum = max(low.max(), high.max()) * 1.05
    ax.plot([0, maximum], [0, maximum], '--', color='black', label='Equal error')
    ax.axhline(tolerance, color='gray', linestyle=':'); ax.axvline(tolerance, color='gray', linestyle=':')
    ax.set_xlabel('Error at 100 points'); ax.set_ylabel('Error at 200 points')
    ax.set_title('Below diagonal: 200 improves reference agreement'); ax.legend()
    for ax in axes.ravel():
        ax.grid(axis='y', alpha=.15)
    fig.suptitle(f"Real-data AUC ablation: {summary['saved_maps']} maps, {summary['curve_pairs']} insertion/deletion pairs")
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(output / 'auc_study_overview.png', dpi=180); plt.close(fig)
    return summary, scope_table


def summarize_budget(home, manifest, output):
    studies, pending = [], []
    for task in manifest['tasks']:
        if task['type'] != 'budget':
            continue
        directory = locate(home, manifest, task)
        if not (directory / 'study.json').exists() or not (directory / 'comparisons.json').exists():
            pending.append(task['id']); continue
        study = load_study(directory)
        if study.get('state') != 'complete':
            pending.append(task['id']); continue
        study['path'] = task['output']
        studies.append(study)
    if pending:
        return {'state': 'running', 'completed_pilots': len(studies), 'pending': pending}
    if len(studies) != 4:
        raise ValueError('Expected all four MS-COCO pilots')
    combined = output / 'budget_common'
    combined.mkdir(parents=True, exist_ok=True)
    selection = write_reports(combined, studies)
    table = []
    local_outputs = {task['output']: locate(home, manifest, task) for task in manifest['tasks']}
    for study in studies:
        single = json.loads((local_outputs[study['path']] / 'selection.json').read_text())
        table.append({'runner': study['runner'], 'partitions': Path(study['inputs'][0]['partition_path'][0]).parent.name,
                      'status': single['status'], 'calibration_candidate': single['calibration_candidate'],
                      'selected_budget': single['selected_budget'], 'comparisons': len(study['rows'])})
    pd.DataFrame(table).to_csv(output / 'budget_scope_summary.csv', index=False)
    return {'state': 'complete', 'combined_status': selection['status'],
            'calibration_candidate': selection['calibration_candidate'], 'selected_budget': selection['selected_budget'],
            'comparisons': sum(len(study['rows']) for study in studies), 'pilots': table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_root', type=Path)
    parser.add_argument('--include-auc', action='store_true',
                        help='Also include the separate AUC-resolution study (excluded by default).')
    args = parser.parse_args()
    home = args.study_root.resolve()
    manifest = json.loads((home / 'manifest.json').read_text())
    output = home / 'report'
    output.mkdir(parents=True, exist_ok=True)
    if not args.include_auc:
        budget = summarize_budget(home, manifest, output)
        lines = ['# MS-COCO XAI explanation-budget ablation', '',
            f"Report generated: {datetime.now(timezone.utc).isoformat()}", '',
            'Select one common explanation budget for BPT, BPT+SAM_c, and SAM_c+KernelSHAP. '
            'The swept variable is the explanation budget: 100, 200, 500, 1000, and 2000, '
            'compared with a 4000-budget approximate reference.', '',
            'Four pilots use YOLO11s and DETR with SAM1 and SAM2 partitions on the same five MS-COCO images '
            '(three calibration and two held-out validation). AUC resolution stays fixed at 200 as an '
            'evaluation setting; it is not an ablation variable.', '']
        if budget['state'] == 'complete':
            lines += [f"Combined status: **{budget['combined_status']}**. "
                f"Calibration candidate: {budget['calibration_candidate']}; "
                f"validated common budget: **{budget['selected_budget']}**.", '',
                '[Budget conclusion](budget_common/conclusion.md)', '',
                pd.DataFrame(budget['pilots']).to_markdown(index=False), '',
                '![Budget comparisons](budget_common/budget_curves.png)']
        else:
            lines += [f"Completed pilots: **{budget['completed_pilots']}/4**.", '',
                f"Pending when this report was retrieved: {', '.join(budget['pending'])}.", '',
                'No common budget recommendation is issued until all four pilots finish.']
        lines += ['', 'Selection uses the predeclared map-agreement and AUC-difference criteria. '
            'Held-out failures do not select a larger budget. A failed study is reported honestly without '
            'assuming that 1000 or 2000 is adequate.', '',
            'ShapBPT is deterministic for a fixed image/tree; its repeats check reproducibility. '
            'KernelSHAP uses independent seeded references. This small pilot cannot prove a universal minimum.', '']
        (output / 'analysis.md').write_text('\n'.join(lines))
        (output / 'study_summary.json').write_text(json.dumps({'budget': budget}, indent=2)+'\n')
        print(output / 'analysis.md')
        print(json.dumps({'budget': budget}, indent=2))
        return
    auc, scopes = summarize_auc(home, manifest, output)
    budget = summarize_budget(home, manifest, output)
    (output / 'study_summary.json').write_text(json.dumps({'auc': auc, 'budget': budget}, indent=2)+'\n')
    lines = ['# Real-data ablation analysis', '',
        f"Report generated: {datetime.now(timezone.utc).isoformat()}", '',
        '## AUC: 100 versus 200 points', '',
        f"Across {auc['saved_maps']} saved attribution maps ({auc['image_cases']} model/background image cases, "
        f"{auc['unique_source_images']} unique images, {auc['scopes']} setups), 200 points reduced mean error "
        f"against the 1000-point reference by **{auc['mean_error_reduction_percent']:.1f}%** "
        f"({auc['mean_error_100']:.6f} → {auc['mean_error_200']:.6f}). "
        f"The measured candidate evaluation time rose **{(auc['runtime_ratio']-1)*100:.1f}%** "
        f"({auc['seconds_100']:.1f}s → {auc['seconds_200']:.1f}s summed median curve times).", '',
        f"At the predeclared {auc['tolerance']:.3f} tolerance, **{auc['within_tolerance_100']}/{auc['curve_pairs']}** "
        f"curves passed at 100 points and **{auc['within_tolerance_200']}/{auc['curve_pairs']}** at 200. "
        f"200 improved {auc['improved']} curves, worsened {auc['worsened']}, and left {auc['unchanged']} unchanged. "
        f"Worst reference error was {auc['max_error_100']:.6f} at 100 and {auc['max_error_200']:.6f} at 200.", '',
        '**Practical conclusion:** keep 200 as the provisional common default for better mean precision. '
        'This study does not validate either resolution uniformly across every tested map. '
        'The per-scope decisions remain governed by the predeclared criteria; the scope CSV lists them. '
        'The reference is approximate, and this is a small convenience sample, not a population-level guarantee.', '',
        f"There were {auc['pairwise_order_or_tie_changes']} method-order/tie changes among "
        f"{auc['pairwise_method_order_comparisons']} within-image/direction comparisons. "
        f"{auc['pairwise_changes_with_gap_at_least_tolerance']} had a gap of at least {auc['tolerance']:.3f} at either resolution. "
        'This margin check is an exploratory explanation of near ties, not a replacement for the original strict ranking criterion.', '',
        '![AUC comparison](auc_study_overview.png)', '',
        scopes[['dataset','runner','background','images','methods','mean_error_100','mean_error_200','runtime_ratio']].to_markdown(index=False, floatfmt='.6f'), '',
        '## MS-COCO explanation-budget study', '']
    if budget['state'] != 'complete':
        lines += [f"Still running: {', '.join(budget['pending'])}. No budget recommendation is issued yet."]
    else:
        lines += [f"Combined status: **{budget['combined_status']}**. Calibration candidate: "
                  f"{budget['calibration_candidate']}; validated common budget: **{budget['selected_budget']}**.", '',
                  '[Full budget conclusion](budget_common/conclusion.md)', '',
                  pd.DataFrame(budget['pilots']).to_markdown(index=False), '',
                  '![Budget comparisons](budget_common/budget_curves.png)']
    lines += ['', '## Verification and limits', '',
        f"Performed {auc['audited_curve_integrations']} integration checks on {auc['distinct_saved_curves']} distinct saved curves "
        f"(reference curves are checked against both candidate rows); maximum discrepancy "
        f"from reported AUC was {auc['max_auc_integration_discrepancy']:.3g}. Candidate AUC repeat range was "
        f"{auc['max_candidate_repeat_range']:.3g}. Source maps were frozen before inference. "
        "All timing ratios exclude model loading, explanation generation, endpoint warm-up, and reference evaluation. "
        "Two GPU workers ran concurrently; CPU/memory contention is part of this hardware-specific measurement.", '',
        'Pooled AUC means weight curves equally, rather than balancing datasets. Methods, backgrounds, and models '
        'can share an image; these curves are correlated observations, not independent population samples.', '',
        'The budget study uses five shared MS-COCO images, three calibration and two validation, '
        'two detectors and two SAM variants, independent seeded KernelSHAP references, and a 4000-budget reference. '
        'ShapBPT is deterministic for a fixed image/tree: its repeated maps test computational reproducibility, '
        'not independent Monte Carlo draws. AUC resolution remains fixed at 200 as declared; '
        'the AUC-resolution findings above do not silently change the budget-selection rule.', '']
    text = '\n'.join(lines)
    (output / 'analysis.md').write_text(text)
    print(output / 'analysis.md')
    print(json.dumps({'auc': auc, 'budget': budget}, indent=2))


if __name__ == '__main__':
    main()
