#!/usr/bin/env python3
"""Which XAI method wins: in one finished run, or across every dataset and domain.

One run (also written automatically at the end of every explanation run):
    python src/scripts/explanations/xai_leaderboard.py dataset results/evaluations/COD_ViT/<partitions>/bg_full_B1000

Every finished run under results/evaluations:
    python src/scripts/explanations/xai_leaderboard.py all --budget 1000 --bg-type full

Methods are ranked within each image (1 = best), so images with different confidence and
datasets with different models count equally. Within a run, the winner of a metric is the
method with the best mean rank; a Wilcoxon test against every other method (Holm-corrected)
says whether the lead is clear. Across datasets, each dataset's mean ranks are averaged, with
every dataset counted once, following Demsar (JMLR 2006).
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

TABLE = 'auc_results_all_images_current_run.csv'
# metric: (better, group, label)
METRICS = {
    'auc_ins': ('higher', 'faithfulness', 'Insertion AUC'),
    'auc_del': ('lower', 'faithfulness', 'Deletion AUC'),
    'road_gap': ('higher', 'faithfulness', 'ROAD gap'),
    'road_morf': ('lower', 'faithfulness', 'ROAD MoRF'),
    'road_lerf': ('higher', 'faithfulness', 'ROAD LeRF'),
    'best_iou': ('higher', 'localization', 'Best IoU with GT'),
    'pointing_accuracy': ('higher', 'localization', 'Pointing game'),
    'sensitivity_n_corr': ('higher', 'robustness', 'Sensitivity-n'),
    'stability_top10_jaccard': ('higher', 'robustness', 'Top-10% stability'),
    'time_exp': ('lower', 'cost', 'Explanation time (s)'),
}
# The overall rank averages these; the others are reported beside it.
OVERALL = ['auc_ins', 'auc_del', 'road_gap']
DOMAINS = {'ImageNet': 'natural images', 'MSCOCO': 'natural images', 'PartImageNet': 'object parts',
           'PACO': 'object parts', 'FunnyBirds': 'synthetic', 'CAMO': 'camouflage', 'CelebAHQ': 'faces',
           'BUSI': 'medical', 'KvasirSEG': 'medical', 'ISIC2018': 'medical'}
ALPHA = .05
FEW_IMAGES = 10
RUN_FOLDER = re.compile(r'bg_(?P<bg>\w+?)_B(?P<budget>\d+)$')

# Plot ink from the reference palette: one neutral box colour, the winner in blue.
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
NEUTRAL, ACCENT = '#c9c8c1', '#2a78d6'
RANK_RAMP = ['#104281', '#256abf', '#5598e7', '#9ec5f4', '#e6effb']  # Rank 1 darkest.


# ---------------------------------------------------------------- one run

def read_table(path):
    frame = pd.read_csv(path, dtype={'image_id': str, 'image_no': str})
    return use_relative_road(frame)


def use_relative_road(frame):
    """Prefer ROAD scores relative to the unmodified image, when every image has them."""
    frame = frame.copy()
    frame.attrs['road'] = 'raw'
    for metric in ['road_gap', 'road_morf', 'road_lerf']:
        relative = metric + '_rel'
        if metric in frame and relative in frame and frame[metric].notna().any():
            if frame.loc[frame[metric].notna(), relative].notna().all():
                frame[metric] = frame[relative]
                frame.attrs['road'] = 'relative'
    return frame


def metrics_in(frame):
    return [metric for metric in METRICS if metric in frame and frame[metric].notna().any()]


def method_order(frame):
    return list(dict.fromkeys(frame['method']))


def paired(frame, metric, methods=None):
    """image x method values, keeping only images where every method has a value."""
    table = frame.pivot_table(index='image_id', columns='method', values=metric, aggfunc='mean')
    table = table[[method for method in (methods or method_order(frame)) if method in table]]
    return table.dropna(axis=1, how='all').dropna()


def ranks_of(table, metric):
    return table.rank(axis=1, ascending=METRICS[metric][0] == 'lower', method='average')


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.
    for position, index in enumerate(order):
        running = max(running, min(1., (len(pvalues) - position) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def wilcoxon_p(first, second):
    from scipy.stats import wilcoxon
    difference = np.asarray(first) - np.asarray(second)
    if len(difference) < 2 or np.allclose(difference, 0):
        return 1.
    return float(wilcoxon(difference).pvalue)


def friedman_p(ranks_table):
    from scipy.stats import friedmanchisquare
    if ranks_table.shape[0] < 2 or ranks_table.shape[1] < 3:
        return None
    if np.allclose(ranks_table.values, ranks_table.values[:1]):
        return 1.
    return float(friedmanchisquare(*[ranks_table[column] for column in ranks_table]).pvalue)


def metric_winner(table, ranks, metric):
    mean = ranks.mean().sort_values()
    winner = mean.index[0]
    others = list(mean.index[1:])
    adjusted = holm([wilcoxon_p(table[winner], table[other]) for other in others]) if others else []
    tied = [other for other, p in zip(others, adjusted) if p >= ALPHA]
    return {'metric': metric, 'better': METRICS[metric][0], 'group': METRICS[metric][1],
            'images': len(table), 'winner': winner, 'winner_mean_rank': float(mean.iloc[0]),
            'runner_up': others[0] if others else None,
            'runner_up_mean_rank': float(mean.iloc[1]) if others else None,
            'friedman_p': friedman_p(ranks), 'clear_winner': bool(others) and not tied,
            'not_separable_from': ', '.join(tied)}


def leaderboard(frame):
    """Per-method mean, median, mean rank and win share for every metric, plus the overall rank."""
    methods, rows, winners = method_order(frame), {}, []
    for metric in metrics_in(frame):
        table = paired(frame, metric, methods)
        if table.shape[1] < 2 or table.empty:
            continue
        ranks = ranks_of(table, metric)
        winners.append(metric_winner(table, ranks, metric))
        best = ranks.eq(ranks.min(axis=1), axis=0)
        for method in table:
            row = rows.setdefault(method, {'method': method})
            row[f'{metric}_mean'] = float(table[method].mean())
            row[f'{metric}_median'] = float(table[method].median())
            row[f'{metric}_rank'] = float(ranks[method].mean())
            row[f'{metric}_win_share'] = float(best[method].mean())
    board = pd.DataFrame([rows[method] for method in methods if method in rows])
    used = [metric for metric in OVERALL if f'{metric}_rank' in board]
    board['overall_rank'] = board[[f'{metric}_rank' for metric in used]].mean(axis=1) if used else np.nan
    board = board.sort_values('overall_rank', kind='stable').reset_index(drop=True)
    board.insert(1, 'position', range(1, len(board) + 1))
    return board, pd.DataFrame(winners), used


def conclusion(board, winners, used, frame, title):
    images = frame['image_id'].nunique()
    lines = [f'# Who wins: {title}', '',
             f"{images} images, {len(board)} methods. Rank 1 is best; ranks are computed within each image."]
    if images < FEW_IMAGES:
        lines += ['', f'**Preview only:** fewer than {FEW_IMAGES} images, so no lead can be called clear yet.']
    if used:
        top = board.head(3)
        names = ', '.join(f"{row.method} ({row.overall_rank:.2f})" for row in top.itertuples())
        lines += ['', f"**Overall** (mean rank over {', '.join(used)}): {names}."]
    else:
        lines += ['', '**Overall:** none of ' + ', '.join(OVERALL) + ' was computed, so there is no overall rank.']
    if frame.attrs.get('road') == 'relative':
        lines += ['', 'ROAD scores are relative to the unmodified image.']
    if not winners.empty:
        lines += ['', '| Metric | Better | Winner | Mean rank | Runner-up | Lead |', '| --- | --- | --- | --- | --- | --- |']
        for row in winners.itertuples():
            tied = row.not_separable_from.split(', ') if isinstance(row.not_separable_from, str) and row.not_separable_from else []
            if row.clear_winner:
                lead = 'clear (p < .05 vs every other)'
            elif len(tied) >= len(board) - 1:
                lead = 'no clear difference between any methods'
            elif tied:
                lead = 'not separable from ' + ', '.join(tied[:3]) + (f' and {len(tied) - 3} more' if len(tied) > 3 else '')
            else:
                lead = ''
            runner = f'{row.runner_up} ({row.runner_up_mean_rank:.2f})' if row.runner_up else ''
            lines.append(f"| {METRICS[row.metric][2]} | {row.better} | {row.winner} | {row.winner_mean_rank:.2f} "
                         f"| {runner} | {lead} |")
    return '\n'.join(lines) + '\n'


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def plot_boxplots(frame, board, winners, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [metric for metric in metrics_in(frame) if metric in set(winners.get('metric', []))]
    if not metrics:
        return None
    order = list(board['method'])
    columns = min(3, len(metrics))
    rows = -(-len(metrics) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.9 * rows), squeeze=False, facecolor=SURFACE)
    winner = dict(zip(winners['metric'], winners['winner']))
    for ax, metric in zip(axes.ravel(), metrics):
        style(ax)
        table = paired(frame, metric, order)
        present = [method for method in order if method in table]
        parts = ax.boxplot([table[method] for method in present], patch_artist=True, widths=.55,
                           medianprops={'color': INK, 'linewidth': 1.6},
                           whiskerprops={'color': MUTED}, capprops={'color': MUTED},
                           flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': MUTED,
                                       'markeredgecolor': SURFACE, 'alpha': .6})
        for patch, method in zip(parts['boxes'], present):
            patch.set_facecolor(ACCENT if method == winner[metric] else NEUTRAL)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(2)
        ax.set_xticks(range(1, len(present) + 1), present, rotation=40, ha='right')
        arrow = '↑ higher is better' if METRICS[metric][0] == 'higher' else '↓ lower is better'
        ax.set_title(f'{METRICS[metric][2]}\n{arrow}', fontsize=10, color=INK, loc='left')
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(f'{frame["image_id"].nunique()} images. Blue: best mean rank. Methods ordered by overall rank.',
                 fontsize=9, color=MUTED, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_rank_heatmap(values, path, title, row_label):
    """values: rows x columns of mean ranks; rank 1 darkest."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    if values.empty:
        return None
    worst = max(2., float(np.nanmax(values.values)))
    fig, ax = plt.subplots(figsize=(1.0 + .95 * values.shape[1], 1.2 + .42 * values.shape[0]), facecolor=SURFACE)
    image = ax.imshow(values.values, cmap=LinearSegmentedColormap.from_list('rank', RANK_RAMP),
                      vmin=1, vmax=worst, aspect='auto')
    for (row, column), value in np.ndenumerate(values.values):
        if np.isfinite(value):
            dark = (value - 1) / (worst - 1) < .5
            ax.text(column, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                    color='#ffffff' if dark else INK)
    ax.set_xticks(range(values.shape[1]), values.columns, rotation=40, ha='right', fontsize=8, color=INK)
    ax.set_yticks(range(values.shape[0]), values.index, fontsize=8, color=INK)
    ax.set_ylabel(row_label, color=MUTED, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=.04, pad=.02)
    bar.set_label('mean rank (1 = best)', color=MUTED, fontsize=8)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def dataset_report(frame, output_dir, title='this run', save_plots=True):
    """Write leaderboard.csv, metric_winners.csv, winner.md and the plots for one run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = frame if 'road' in frame.attrs else use_relative_road(frame)
    if frame.empty or frame['method'].nunique() < 2:
        return None
    board, winners, used = leaderboard(frame)
    board.to_csv(output_dir / 'leaderboard.csv', index=False)
    winners.to_csv(output_dir / 'metric_winners.csv', index=False)
    (output_dir / 'winner.md').write_text(conclusion(board, winners, used, frame, title))
    if save_plots:
        plot_boxplots(frame, board, winners, output_dir / 'method_boxplots.png')
        ranks = board.set_index('method')[[f'{metric}_rank' for metric in metrics_in(frame)
                                            if f'{metric}_rank' in board]]
        ranks.columns = [METRICS[column[:-5]][2] for column in ranks.columns]
        plot_rank_heatmap(ranks, output_dir / 'method_ranks.png', f'Mean rank per metric: {title}', 'method')
    return {'board': board, 'winners': winners, 'overall_metrics': used}


# ---------------------------------------------------------------- every run

def find_runs(root, budget=None, bg_type=None):
    runs = []
    for path in sorted(Path(root).glob(f'*/*/bg_*_B*/{TABLE}')):
        match = RUN_FOLDER.match(path.parent.name)
        if not match:
            continue
        if budget is not None and int(match['budget']) != budget:
            continue
        if bg_type is not None and match['bg'] != bg_type:
            continue
        partitions = path.parent.parent.name
        dataset = partitions.split('_sam')[0]
        runs.append({'runner': path.parent.parent.parent.name, 'partitions': partitions, 'dataset': dataset,
                     'domain': DOMAINS.get(dataset, 'other'), 'bg_type': match['bg'],
                     'budget': int(match['budget']), 'table': path})
    return runs


def shared_methods(frames, requested=None):
    """The methods to compare: requested ones, else those found in at least half of the runs."""
    if requested:
        return list(requested)
    counts = {}
    for frame in frames:
        for method in method_order(frame):
            counts[method] = counts.get(method, 0) + 1
    return [method for method, count in counts.items() if count >= len(frames) / 2]


def run_ranks(frame, methods):
    """Mean per-image rank of each method, per metric, within one run."""
    rows = []
    for metric in metrics_in(frame):
        table = paired(frame, metric, methods)
        if list(table.columns) != methods or table.empty:
            continue
        for method, rank in ranks_of(table, metric).mean().items():
            rows.append({'metric': metric, 'method': method, 'mean_rank': float(rank), 'images': len(table)})
    return rows


def all_report(root, output_dir, budget=None, bg_type=None, methods=None, save_plots=True):
    runs = find_runs(root, budget, bg_type)
    if not runs:
        raise FileNotFoundError(f'No {TABLE} under {root}/<runner>/<partitions>/bg_*_B*')
    frames = [read_table(run['table']) for run in runs]
    selected = shared_methods(frames, methods)
    if len(selected) < 2:
        raise ValueError('Fewer than two methods are shared by the runs; pass --methods or --budget')
    long, kept = [], []
    for run, frame in zip(runs, frames):
        missing = [method for method in selected if method not in set(frame['method'])]
        run.update({'images': frame['image_id'].nunique(), 'missing_methods': ', '.join(missing)})
        if missing:
            run['included'] = False
            continue
        rows = run_ranks(frame[frame['method'].isin(selected)], selected)
        run['metrics'] = ', '.join(dict.fromkeys(row['metric'] for row in rows))
        run['included'] = bool(rows)
        label = f"{run['dataset']} | {run['runner']} | {run['partitions'][len(run['dataset']) + 1:]}"
        long += [{**row, 'run': label, **{key: run[key] for key in ['dataset', 'domain', 'runner', 'budget', 'bg_type']}}
                 for row in rows]
        kept.append(run)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{key: value for key, value in run.items() if key != 'table'} | {'table': str(run['table'])}
                  for run in runs]).to_csv(output_dir / 'runs.csv', index=False)
    long = pd.DataFrame(long)
    if long.empty:
        raise ValueError('No run has every selected method; see runs.csv')
    long.to_csv(output_dir / 'run_ranks.csv', index=False)
    # Several runs of one dataset (other model or partitions) count once, as their mean.
    per_dataset = long.groupby(['domain', 'dataset', 'metric', 'method'], as_index=False)['mean_rank'].mean()
    overall_used = [metric for metric in OVERALL if metric in set(per_dataset['metric'])]
    composite = (per_dataset[per_dataset['metric'].isin(overall_used)]
                 .groupby(['domain', 'dataset', 'method'], as_index=False)['mean_rank'].mean())
    dataset_table = composite.pivot_table(index=['domain', 'dataset'], columns='method', values='mean_rank')[selected]
    dataset_table.to_csv(output_dir / 'dataset_ranks.csv')
    domain_table = composite.groupby(['domain', 'method'])['mean_rank'].mean().unstack()[selected]
    domain_table.to_csv(output_dir / 'domain_ranks.csv')
    overall = pd.DataFrame({'method': selected})
    complete = dataset_table.dropna()
    overall['overall_rank'] = [complete[method].mean() for method in selected]
    overall['datasets_won'] = [int((complete.idxmin(axis=1) == method).sum()) for method in selected]
    overall['domains_won'] = [int((domain_table.dropna().idxmin(axis=1) == method).sum()) for method in selected]
    metric_table = per_dataset.groupby(['metric', 'method'])['mean_rank'].mean().unstack()
    for metric in [metric for metric in METRICS if metric in metric_table.index]:
        overall[f'{metric}_rank'] = [metric_table.loc[metric].get(method) for method in selected]
    overall = overall.sort_values('overall_rank', kind='stable').reset_index(drop=True)
    overall.insert(1, 'position', range(1, len(overall) + 1))
    overall.to_csv(output_dir / 'overall.csv', index=False)
    test = friedman_p(complete) if len(complete) >= 2 else None
    (output_dir / 'winner.md').write_text(all_conclusion(overall, complete, domain_table, kept, runs, overall_used, test))
    if save_plots:
        ordered = list(overall['method'])
        table = dataset_table[ordered].copy()
        table.index = [f'{dataset} ({domain})' for domain, dataset in table.index]
        plot_rank_heatmap(table, output_dir / 'dataset_ranks.png',
                          f"Overall rank per dataset ({', '.join(overall_used)})", 'dataset')
        plot_rank_heatmap(domain_table[ordered], output_dir / 'domain_ranks.png', 'Overall rank per domain', 'domain')
        plot_rank_boxplots(per_dataset, ordered, output_dir / 'rank_boxplots.png')
    return {'overall': overall, 'datasets': dataset_table, 'domains': domain_table, 'runs': runs}


def all_conclusion(overall, complete, domain_table, kept, runs, used, test):
    lines = ['# Who wins across datasets', '',
             f"{len(complete)} datasets ({len(kept)} of {len(runs)} runs), {len(overall)} methods, "
             f"overall rank = mean rank over {', '.join(used)}, each dataset counted once."]
    skipped = [run for run in runs if not run.get('included')]
    if skipped:
        lines += ['', f"Left out {len(skipped)} runs without every compared method: "
                  + '; '.join(f"{run['dataset']} {run['runner']} ({run['missing_methods'] or 'no metrics'})"
                              for run in skipped[:6]) + ('...' if len(skipped) > 6 else '')]
    lines += ['', '| Position | Method | Overall rank | Datasets won | Domains won |', '| --- | --- | --- | --- | --- |']
    lines += [f"| {row.position} | {row.method} | {row.overall_rank:.2f} | {row.datasets_won} | {row.domains_won} |"
              for row in overall.itertuples()]
    if test is not None:
        verdict = 'the methods differ' if test < ALPHA else 'no significant difference between the methods yet'
        lines += ['', f'Friedman test over datasets: p = {test:.3g} ({verdict}).']
    if len(complete) < 5:
        lines += ['', 'With fewer than 5 datasets, read this as a preview.']
    lines += ['', '## Winner per domain', '']
    for domain, row in domain_table.iterrows():
        lines.append(f"- **{domain}:** {row.idxmin()} (mean rank {row.min():.2f})")
    return '\n'.join(lines) + '\n'


def plot_rank_boxplots(per_dataset, ordered, path):
    """Per metric, the spread of each method's mean rank over datasets."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [metric for metric in METRICS if metric in set(per_dataset['metric'])]
    if not metrics:
        return None
    columns = min(3, len(metrics))
    rows = -(-len(metrics) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.9 * rows), squeeze=False, facecolor=SURFACE)
    for ax, metric in zip(axes.ravel(), metrics):
        style(ax)
        table = per_dataset[per_dataset['metric'] == metric].pivot_table(index='dataset', columns='method',
                                                                          values='mean_rank')
        present = [method for method in ordered if method in table]
        best = table[present].mean().idxmin()
        parts = ax.boxplot([table[method].dropna() for method in present], patch_artist=True, widths=.55,
                           medianprops={'color': INK, 'linewidth': 1.6}, whiskerprops={'color': MUTED},
                           capprops={'color': MUTED}, showfliers=False)
        for patch, method in zip(parts['boxes'], present):
            patch.set_facecolor(ACCENT if method == best else NEUTRAL)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(2)
        for position, method in enumerate(present, start=1):
            values = table[method].dropna()
            ax.scatter(np.full(len(values), position), values, s=14, color=INK, edgecolor=SURFACE,
                       linewidth=.8, zorder=3)
        ax.set_xticks(range(1, len(present) + 1), present, rotation=40, ha='right')
        ax.set_ylim(len(present) + .4, .6)  # Same scale in every panel, rank 1 at the top.
        ax.set_yticks(range(1, len(present) + 1))
        ax.set_ylabel('mean rank (1 = best, top)', fontsize=8, color=MUTED)
        ax.set_title(METRICS[metric][2], fontsize=10, color=INK, loc='left')
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle('One dot per dataset. Blue: best mean rank over datasets.', fontsize=9, color=MUTED, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- command line

def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest='command', required=True)
    one = commands.add_parser('dataset', help='Rank the methods of one finished run')
    one.add_argument('folder', type=Path, help=f'Evaluation or explanation folder holding {TABLE}')
    one.add_argument('--output-dir', type=Path, help='Default: the folder itself')
    one.add_argument('--no-plots', dest='save_plots', action='store_false')
    every = commands.add_parser('all', help='Rank the methods across every finished run')
    every.add_argument('--root', type=Path, default=Path('results/evaluations'))
    every.add_argument('--budget', type=int, help='Only runs with this budget (bg_*_B<budget>)')
    every.add_argument('--bg-type', help='Only runs with this background (bg_<type>_B*)')
    every.add_argument('--methods', nargs='+', help='Methods to compare; default: those in at least half the runs')
    every.add_argument('--output-dir', type=Path, help='Default: <root>/xai_leaderboard')
    every.add_argument('--no-plots', dest='save_plots', action='store_false')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'dataset':
        folder = args.folder
        title = ' / '.join(folder.resolve().parts[-3:])
        result = dataset_report(read_table(folder / TABLE), args.output_dir or folder, title, args.save_plots)
        if result is None:
            raise ValueError(f'{folder / TABLE}: needs at least two methods')
        print((Path(args.output_dir or folder) / 'winner.md').read_text())
        return result
    output = args.output_dir or args.root / 'xai_leaderboard'
    result = all_report(args.root, output, args.budget, args.bg_type, args.methods, args.save_plots)
    print((output / 'winner.md').read_text())
    print(f'Saved: {output}')
    return result


if __name__ == '__main__':
    main()
