#!/usr/bin/env python3
"""Run BPT/BPT+SAM_c budget checkpoints interactively, retaining a fixed hold-out."""
import argparse
import csv
import json
from pathlib import Path


def statistical_report(folder, study):
    from budget_noninferiority import analyze
    folder = Path(folder)
    if min(map(len, study['splits'].values())) < 2:
        result = {'status': 'insufficient_images', 'selected_budget': None,
                  'claim': 'At least two images per split are needed even to compute this bootstrap; use a larger fixed pilot.'}
    else:
        result = analyze([dict(study, path=str(folder))], study['statistical_protocol'])
    conclusion = (f"Recommended tested budget: {result['selected_budget']} for {', '.join(study['methods'])}."
        if result.get('selected_budget') is not None else
        'No common budget is validated for the two tested methods. Do not automatically choose the largest tested budget.')
    (folder/'statistical_selection.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Paired non-inferiority budget conclusion', '', conclusion, '',
        f"Status: {result['status']}. Methods: {', '.join(study['methods'])}. Background: {study['background']}.", '',
        f"This tests mean agreement with a {study['reference_budget']}-budget approximation, not attribution correctness. "
        'It does not select a budget for KernelSHAP, LIME, gradient methods or untested datasets/models.', '',
        'Images are the bootstrap unit. Repeated deterministic BPT maps are not independent samples. '
        'Calibration checkpoints are descriptive; the fixed held-out set is tested once at the final stage. '
        'A held-out failure rejects the candidate without selecting a larger one from those outcomes.', '']
    bound_rows = []
    for split, output in result.get('splits', {}).items():
        lines += [f'## {split.capitalize()}', '',
                  f"Reference checks passed: {output['reference_stable']}.", '',
                  '| Budget | Worst upper bound / acceptable loss | All metrics pass |', '| ---: | ---: | --- |']
        for budget in study['budgets']:
            group = [row for row in output['bounds'] if row['budget'] == budget]
            worst = max(row['upper_bound']/row['margin'] for row in group)
            lines.append(f"| {budget} | {worst:.4f} | {all(row['pass'] for row in group)} |")
        bound_rows.extend(dict(row, split=split) for row in output['bounds'])
    if bound_rows:
        with (folder/'statistical_bounds.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(bound_rows[0])); writer.writeheader(); writer.writerows(bound_rows)
    (folder/'statistical_conclusion.md').write_text('\n'.join(lines)+'\n')
    print('\nSTATISTICAL DECISION: '+conclusion, flush=True)
    print('Statistical report:', folder/'statistical_conclusion.md', flush=True)
    return result


def build_args(source_folder, images=3, final=False, study_name='interactive_bpt_r8000',
               budgets=None, reference_budget=8000):
    import budget_ablation as pilot
    source_folder = Path(source_folder).resolve()
    source = json.loads((source_folder/'study.json').read_text())
    if source.get('runner') != 'OD_yolo' or source.get('background') != 'full':
        raise ValueError('Use the existing MS-COCO YOLO full-background pilot as the source')
    if 'sam1' not in source_folder.parent.parent.name.lower():
        raise ValueError('This interactive workflow is currently scoped to SAM1')
    args = pilot.parser(['--runner', source['runner']])
    for key, value in source['settings'].items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.methods = list(pilot.DEFAULT_METHODS)
    args.budgets = list(pilot.DEFAULT_BUDGETS if budgets is None else budgets)
    args.reference_budget = reference_budget
    args.study_name = study_name
    args.reuse_study = source_folder
    args.progress_every_image = True
    args.statistical_test = True
    args.stop_after_images = None if final else images
    args.resume, args.dry_run = True, False
    args.torch_threads = 6
    calibration = len(source['splits']['calibration'])
    if not final and not 1 <= images <= calibration:
        raise ValueError(f'Use --images between 1 and {calibration}; --final processes the fixed held-out set')
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-study', required=True, type=Path)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--images', type=int, default=3, help='First N original calibration images; default: 3')
    stage.add_argument('--final', action='store_true', help='Finish the original fixed design and perform held-out testing')
    parser.add_argument('--study-name', default='interactive_bpt_r8000')
    parser.add_argument('--budgets', nargs='+', type=int, help='Candidate budgets; default: 1000 2000 4000 6000')
    parser.add_argument('--reference-budget', type=int, default=8000)
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    args = build_args(cli.source_study, cli.images, cli.final, cli.study_name,
                      cli.budgets, cli.reference_budget)
    args.dry_run = cli.dry_run
    print('Methods: BPT, BPT+SAM_c. KernelSHAP excluded. Background: full.', flush=True)
    print('Stage: '+('final fixed-design validation' if cli.final else f'first {cli.images} calibration images; descriptive checkpoint'), flush=True)
    import budget_ablation as pilot
    pilot.run(args)


if __name__ == '__main__':
    main()
