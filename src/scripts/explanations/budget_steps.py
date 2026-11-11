#!/usr/bin/env python3
"""How much do AUC+ and AUC- still change between consecutive budgets?

Each image is explained at every budget and scored with insertion (AUC+) and deletion (AUC-).
The step from one budget to the next is |AUC(next) - AUC(previous)| for that image, averaged
over images. Unlike budget_ablation.py, this needs no high-budget reference map.
Repeat the command to resume; add budgets to the list to reuse the finished ones.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

import budget_ablation as ablation
from budget_selection import DEFAULT_CRITERIA
from config_paths import output_path, resolve_path
import experiment_common as common

DEFAULT_BUDGETS = [20, 50, 100, 500, 1000, 2000]
DEFAULT_IMAGES = 5
METRICS = {'auc_ins': 'AUC+', 'auc_del': 'AUC-'}
STYLES = {'BPT': ('#2a78d6', 'o'), 'BPT+SAM_c': ('#eb6834', 's'), 'SAM_c+KernelSHAP': ('#1baf7a', '^')}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e6e5e0'


def parser(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--runner', choices=ablation.RUNNERS, default='OD_yolo')
    runner = preliminary.parse_known_args(argv)[0].runner
    result = common.create_parser('OD_DETR' if runner == 'OD_yolo' else runner, auc_background=False, road=False)
    result.description = __doc__
    result.add_argument('--runner', choices=ablation.RUNNERS, default=runner)
    result.add_argument('--budgets', '--budget-steps', nargs='+', type=int, default=DEFAULT_BUDGETS,
                        help='Budgets to test; each step is one budget to the next')
    result.add_argument('--methods', nargs='+', choices=ablation.METHODS, default=ablation.DEFAULT_METHODS)
    result.add_argument('--seed', type=int, default=0, help='Selects the images and seeds KernelSHAP')
    result.add_argument('--tolerance', type=float, default=DEFAULT_CRITERIA['max_auc_difference'],
                        help='A step counts as settled when its mean change is at most this')
    result.add_argument('--study-name', default='steps')
    result.set_defaults(bg_type='full')
    if runner == 'OD_yolo':
        result.set_defaults(model='yolo11s', config='MSCOCO_epito_sam1')
    return result


def validate(args):
    if args.max_evals != parser([]).get_default('max_evals'):
        raise ValueError('--budget sets one budget, which this study does not use; pass --budgets 20 50 100')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    args.budgets = sorted(set(args.budgets))
    args.methods = list(dict.fromkeys(args.methods))
    if len(args.budgets) < 2 or args.budgets[0] < 2:
        raise ValueError('Use at least two budgets, each at least 2')
    if not args.compute_auc:
        raise ValueError('This study measures AUC, so --no-auc cannot be used')
    if args.tolerance < 0:
        raise ValueError('--tolerance must be nonnegative')


def select_images(records, limit, seed, explicit):
    """Listed images keep their order. Otherwise a larger limit extends the same random sample."""
    if explicit:
        return records[:limit]
    order = np.random.default_rng(seed).permutation(len(records))[:limit or DEFAULT_IMAGES]
    return [records[int(index)] for index in order]


def step_frame(results):
    """One row per image, method, and consecutive pair of budgets."""
    import pandas as pd
    rows = []
    for (image_id, method), group in results.groupby(['image_id', 'method'], sort=False):
        group = group.sort_values('budget')
        for (_, before), (_, after) in zip(group.iloc[:-1].iterrows(), group.iloc[1:].iterrows()):
            row = {'image_id': image_id, 'method': method, 'from_budget': int(before['budget']),
                   'to_budget': int(after['budget'])}
            for metric in METRICS:
                row[f'{metric}_change'] = after[metric] - before[metric]
                row[f'{metric}_abs_change'] = abs(row[f'{metric}_change'])
            rows.append(row)
    return pd.DataFrame(rows)


def step_summary(steps):
    columns = [f'{metric}_{kind}' for metric in METRICS for kind in ('abs_change', 'change')]
    grouped = steps.groupby(['method', 'from_budget', 'to_budget'], sort=False)[columns]
    summary = grouped.agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.insert(0, 'images', grouped.size())
    return summary.reset_index().sort_values(['method', 'from_budget'], kind='stable')


def settled_budget(summary, method, tolerance):
    """Smallest budget after which every later step, in both AUCs, is within tolerance.

    None when the last step is still above it, because nothing tested shows the curve has settled.
    """
    selected = summary[summary['method'] == method].sort_values('from_budget')
    above = (selected[[f'{metric}_abs_change_mean' for metric in METRICS]] > tolerance).any(axis=1).to_numpy()
    if above[-1]:
        return None
    return int(selected['to_budget'].iloc[np.nonzero(above)[0][-1]]) if above.any() else int(selected['from_budget'].iloc[0])


def draw(plt, path, series, labels, xlabel, ylabels, title, tolerance=None, rings=None):
    """series maps (metric, method) to means and SDs. One panel per metric, on separate y scales."""
    methods = list(dict.fromkeys(method for _, method in series))
    positions = np.arange(len(labels), dtype=float)
    figure, axes = plt.subplots(1, len(METRICS), figsize=(11, 4.4))
    for ax, metric in zip(axes, METRICS):
        for index, method in enumerate(methods):
            means, deviations = series[(metric, method)]
            color, marker = STYLES[method]
            x = positions + (index - (len(methods) - 1) / 2) * .07  # Keeps error bars from overlapping.
            lower = np.minimum(deviations, means) if tolerance is not None else deviations  # |change| is never below 0.
            ax.errorbar(x, means, yerr=[lower, deviations], color=color, marker=marker, markersize=7, linewidth=2,
                        capsize=4, elinewidth=1.2, markeredgecolor='white', label=method)
            ring = (rings or {}).get(method)
            if ring is not None:
                ax.plot(x[ring], means[ring], marker='o', markersize=17, markerfacecolor='none',
                        markeredgecolor=color, markeredgewidth=1.5, linestyle='none', clip_on=False)
        if tolerance is not None:
            ax.axhline(tolerance, color=MUTED, linestyle=':', linewidth=1.2, label=f'Tolerance {tolerance:g}')
            ax.set_ylim(bottom=0)
        ax.set_xticks(positions, labels, rotation=45, ha='right', rotation_mode='anchor')  # Long budget labels stay readable.
        ax.set_xlabel(xlabel, color=MUTED)
        ax.set_ylabel(ylabels[metric], color=MUTED)
        ax.set_title(METRICS[metric], color=INK, loc='left', fontweight='bold')
        ax.grid(axis='y', color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
    axes[0].legend(frameon=False, labelcolor=MUTED)
    figure.suptitle(title, color=INK, x=.01, ha='left', fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(path, dpi=160, facecolor='white')
    plt.close(figure)


def plot_steps(plt, path, summary, tolerance, settled, title):
    methods = list(dict.fromkeys(summary['method']))
    first = summary[summary['method'] == methods[0]]
    labels = [f'{start}→{end}' for start, end in zip(first['from_budget'], first['to_budget'])]
    series, rings = {}, {}
    for method in methods:
        selected = summary[summary['method'] == method]
        for metric in METRICS:
            series[(metric, method)] = (selected[f'{metric}_abs_change_mean'].to_numpy(),
                                        np.nan_to_num(selected[f'{metric}_abs_change_std'].to_numpy()))
        matches = np.nonzero(selected['from_budget'].to_numpy() == settled[method])[0]
        rings[method] = int(matches[0]) if matches.size else None
    draw(plt, path, series, labels, 'Budget step', {metric: f'E[|Δ{name}|]' for metric, name in METRICS.items()},
         title, tolerance, rings)


def plot_values(plt, path, results, title):
    budgets = sorted(results['budget'].unique())
    series = {}
    for method, group in results.groupby('method', sort=False):
        means = group.groupby('budget')[list(METRICS)].mean()
        deviations = group.groupby('budget')[list(METRICS)].std().fillna(0)
        for metric in METRICS:
            series[(metric, method)] = (means[metric].to_numpy(), deviations[metric].to_numpy())
    draw(plt, path, series, [str(budget) for budget in budgets], 'Budget',
         {metric: f'E[{name}]' for metric, name in METRICS.items()}, title)


def write_reports(runtime, folder, rows, args, partition_name, run_timing=None):
    results = runtime.pd.DataFrame(rows)
    results.to_csv(folder / 'budget_auc.csv', index=False)
    steps = step_frame(results)
    steps.to_csv(folder / 'budget_steps.csv', index=False)
    summary = step_summary(steps)
    summary.to_csv(folder / 'budget_steps_summary.csv', index=False)
    settled = {method: settled_budget(summary, method, args.tolerance) for method in args.methods}
    common.write_json(folder / 'settled_budget.json', {'tolerance': args.tolerance, 'budgets': settled})
    images = results['image_id'].nunique()
    scope = f'{args.runner}, {partition_name}, {images} images. Error bars: SD across images.'
    plot_steps(runtime.plt, folder / 'budget_steps.png', summary, args.tolerance, settled, f'AUC change per budget step. {scope}')
    plot_values(runtime.plt, folder / 'budget_auc.png', results, f'AUC by budget. {scope}')
    for _, row in summary.iterrows():
        print(f"{row['method']:<18} {row['from_budget']:>5} -> {row['to_budget']:<5} "
              f"|dAUC+| {row['auc_ins_abs_change_mean']:.4f}  |dAUC-| {row['auc_del_abs_change_mean']:.4f}", flush=True)
    print(f'Settled at tolerance {args.tolerance:g}: {settled}', flush=True)
    write_timing(runtime, folder, results, run_timing or {}, f'{args.runner}, {partition_name}, {images} images')
    print(f'Saved: {folder}', flush=True)
    return summary


def timing_table(results):
    """Explanation time per method and budget, from the time saved with each map (cached maps included)."""
    grouped = results.groupby(['method', 'budget'], sort=False)
    table = grouped.agg(images=('image_id', 'nunique'), explanation_sec_mean=('explanation_sec', 'mean'),
                        explanation_sec_std=('explanation_sec', 'std'), explanation_sec_total=('explanation_sec', 'sum'),
                        evaluation_sec_mean=('evaluation_sec', 'mean'), mask_evaluations_mean=('mask_evaluations', 'mean'))
    table['explanation_sec_std'] = table['explanation_sec_std'].fillna(0)
    table['ms_per_mask'] = 1000 * table['explanation_sec_mean'] / table['mask_evaluations_mean'].where(
        table['mask_evaluations_mean'] > 0)
    return table.reset_index().sort_values(['method', 'budget'], kind='stable')


def timing_summary(results, table, run_timing):
    methods = {}
    for method, group in results.groupby('method', sort=False):
        trees = group.drop_duplicates('image_id')['tree_sec']
        methods[method] = {'maps': len(group), 'explanation_sec': float(group['explanation_sec'].sum()),
                           'evaluation_sec': float(group['evaluation_sec'].sum(skipna=True)),
                           'tree_sec_this_run': float(trees.sum()),
                           'explanation_sec_at_largest_budget': float(table[table['method'] == method]['explanation_sec_mean'].iloc[-1])}
    computed = results[results['computed']]
    return {
        'note': 'all_maps times are those saved with each map, so cached maps keep the time of the run that made them',
        'this_run': {**run_timing, 'maps_computed': int(len(computed)), 'maps_cached': int((~results['computed']).sum()),
                     'explanation_sec': float(computed['explanation_sec'].sum()),
                     'evaluation_sec': float(computed['evaluation_sec'].sum(skipna=True))},
        'all_maps': {'maps': len(results), 'images': int(results['image_id'].nunique()),
                     'explanation_sec': float(results['explanation_sec'].sum()),
                     'evaluation_sec': float(results['evaluation_sec'].sum(skipna=True)), 'methods': methods}}


def duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, rest = divmod(round(seconds), 60)
    return f'{minutes}m {rest:02d}s' if minutes < 60 else f'{minutes // 60}h {minutes % 60:02d}m'


def plot_time(plt, path, table, title):
    """Mean explanation time per image against budget, one line per method, on a log scale."""
    budgets = sorted(table['budget'].unique())
    positions = {budget: index for index, budget in enumerate(budgets)}
    figure, ax = plt.subplots(figsize=(7.5, 4.6))
    for method, group in table.groupby('method', sort=False):
        color, marker = STYLES.get(method, (MUTED, 'o'))
        x = [positions[budget] for budget in group['budget']]
        means, deviations = group['explanation_sec_mean'].to_numpy(), group['explanation_sec_std'].to_numpy()
        ax.errorbar(x, means, yerr=[np.minimum(deviations, means * .999), deviations], color=color, marker=marker,
                    markersize=7, linewidth=2, capsize=4, elinewidth=1.2, markeredgecolor='white', label=method)
        ax.annotate(duration(means[-1]), (x[-1], means[-1]), xytext=(8, 0), textcoords='offset points',
                    va='center', color=INK, fontsize=9)
    ax.set_yscale('log')
    ax.set_xticks(range(len(budgets)), [str(budget) for budget in budgets], rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlim(-.4, len(budgets) - .4)
    ax.set_xlabel('Budget', color=MUTED)
    ax.set_ylabel('Explanation time per image (s, log scale)', color=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=.8, which='both')
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, which='both')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.legend(frameon=False, labelcolor=MUTED)
    figure.suptitle(title, color=INK, x=.01, ha='left', fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .93))
    figure.savefig(path, dpi=160, facecolor='white')
    plt.close(figure)


def write_timing(runtime, folder, results, run_timing, scope):
    table = timing_table(results)
    table.to_csv(folder / 'budget_timing.csv', index=False)
    summary = timing_summary(results, table, run_timing)
    common.write_json(folder / 'timing_summary.json', summary)
    plot_time(runtime.plt, folder / 'budget_time.png', table,
              f'Explanation time by budget\n{scope}. Error bars: SD across images.')
    print('\nTiming (mean per image; saved times, cached maps included):', flush=True)
    print(f"{'Method':<18} {'Budget':>6} {'Explain':>9} {'SD':>8} {'Score AUC':>10} {'Masks':>7} {'ms/mask':>8}", flush=True)
    for _, row in table.iterrows():
        print(f"{row['method']:<18} {row['budget']:>6} {duration(row['explanation_sec_mean']):>9} "
              f"{row['explanation_sec_std']:>7.1f}s {duration(row['evaluation_sec_mean']):>10} "
              f"{row['mask_evaluations_mean']:>7.0f} {row['ms_per_mask']:>8.2f}", flush=True)
    for method, item in summary['all_maps']['methods'].items():
        print(f"{method:<18} all budgets: explain {duration(item['explanation_sec'])}, "
              f"score AUC {duration(item['evaluation_sec'])} over {item['maps']} maps", flush=True)
    this_run = summary['this_run']
    print(f"This run: {this_run['maps_computed']} maps computed, {this_run['maps_cached']} cached; "
          f"explain {duration(this_run['explanation_sec'])}, score AUC {duration(this_run['evaluation_sec'])}"
          + (f"; model load {duration(this_run['model_load_sec'])}" if 'model_load_sec' in this_run else '')
          + (f"; wall {duration(this_run['wall_sec'])}" if 'wall_sec' in this_run else ''), flush=True)
    print(f"All maps: explain {duration(summary['all_maps']['explanation_sec'])}, "
          f"score AUC {duration(summary['all_maps']['evaluation_sec'])}", flush=True)


def explain_image(runtime, args, adapter, record, inputs, folder, signature):
    image_id = record['image_id']
    image = np.asarray(Image.open(record['image_path']).convert('RGB'))
    _, labels = runtime.partitions.load_refine_partitions(image, str(inputs['partition_dir']), image_id, 'compact')
    replacements = common.backgrounds(image, args.bg_type)
    masked_predict = common.make_masked_predictor(adapter, image, replacements, args.model_batch_size)
    counters = {'masks': 0, 'backgrounds': len(replacements)}

    def predict(masks):
        counters['masks'] += len(masks)
        return masked_predict(masks)

    scores = predict(np.ones((1, *image.shape[:2]), dtype=bool))[0]
    target = int(np.argmax(scores)) if args.target_class is None else args.target_class
    explainer = runtime.shap_bpt.Explainer(predict, image, num_explained_classes=None,
                                           explained_class_ids=[target], verbose=False)
    trees, tree_sec = {}, {}
    for method in args.methods:
        if method != 'SAM_c+KernelSHAP':
            tree_started = time.perf_counter()
            trees[method] = runtime.shap_bpt.build_bpt_from_image(
                image, **({'prebuilt_partitions': labels} if method == 'BPT+SAM_c' else {}))
            tree_sec[method] = time.perf_counter() - tree_started
    shared = {'image_id': image_id, 'target': target, 'class_name': adapter.class_names[target],
              'f_S': float(explainer.base_nuN[0]), 'f_0': float(explainer.base_nu0[0])}
    print(f"{image_id}: explaining {shared['class_name']} (score {shared['f_S']:.3f}, empty {shared['f_0']:.3f})", flush=True)
    rows = []
    for method in args.methods:
        for budget in args.budgets:
            cache = folder / image_id / method.replace('+', '_') / f'B{budget}'
            before = counters['masks']  # Scoring a new map always calls the model; a cached one does not.
            _, saved = ablation.evaluate_map(runtime, args, image, labels, predict, explainer, target, trees,
                                             method, budget, args.seed, cache, signature, counters)
            rows.append({**shared, 'method': method, 'budget': budget, 'auc_ins': saved['auc_ins'],
                         'auc_del': saved['auc_del'], 'explanation_sec': saved['explanation_sec'],
                         'evaluation_sec': saved.get('evaluation_sec', np.nan),
                         'mask_evaluations': saved['mask_evaluations'], 'tree_sec': tree_sec.get(method, 0.),
                         'computed': counters['masks'] > before})
    return rows


def run(args, adapter=None):
    validate(args)
    count, args.limit = args.limit, None  # Select uniformly from eligible partitions, not the first N filenames.
    inputs = common.prepare_inputs('OD_DETR' if args.runner == 'OD_yolo' else args.runner, args)
    records = select_images(inputs['records'], count, args.seed, bool(args.image_ids))
    identity = {'runner': args.runner, 'model': args.model, 'checkpoint': str(args.checkpoint), 'seed': args.seed,
                'background': args.bg_type, 'auc_points': args.auc_num_samples, 'target': args.target_class,
                'aggregation': args.class_aggregation, 'partitions': inputs['partition_dir'].name}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = resolve_path(args.output_dir) if args.output_dir else output_path(inputs['config'], 'evaluations') / 'budget_steps'
    folder = root / args.runner / inputs['partition_dir'].name / f'bg_{args.bg_type}' / args.study_name
    images = [record['image_id'] for record in records]
    print(json.dumps({'folder': str(folder), 'images': images, 'budgets': args.budgets, 'methods': args.methods}, indent=2), flush=True)
    if args.dry_run:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    runtime = common.load_runtime()
    if args.torch_threads:
        runtime.torch.set_num_threads(args.torch_threads)
    device = args.device or inputs['config'].get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if runtime.torch.cuda.is_available() else 'mps' if runtime.torch.backends.mps.is_available() else 'cpu'
    device = runtime.torch.device(device)
    run_started = time.perf_counter()
    if adapter is None:
        adapter = ablation.YoloAdapter(args, device) if args.runner == 'OD_yolo' else common.load_adapter(args.runner, args, device)
    common.write_json(folder / 'study.json', {**identity, 'images': images, 'budgets': args.budgets,
                                              'methods': args.methods, 'tolerance': args.tolerance, 'device': str(device)})
    started, rows, summary = time.perf_counter(), [], None
    model_load_sec = started - run_started
    for number, record in enumerate(records, 1):
        rows.extend(explain_image(runtime, args, adapter, record, inputs, folder, signature))
        print(f'IMAGE DONE {number}/{len(records)} | {time.perf_counter() - started:.0f}s', flush=True)
        run_timing = {'model_load_sec': model_load_sec, 'wall_sec': time.perf_counter() - run_started,
                      'images_done': number, 'images': len(records), 'device': str(device)}
        summary = write_reports(runtime, folder, rows, args, inputs['partition_dir'].name, run_timing)  # Inspect while it runs.
    return summary


if __name__ == '__main__':
    run(parser().parse_args())
