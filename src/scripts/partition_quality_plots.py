"""Plots for evaluate_partitions.py, following the GT figures in notebooks/stats.ipynb."""
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from partition_diagnosis import CAUSES, SIZES

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
PARTITION, BASELINE = '#2a78d6', '#eb6834'
LEVELS = [('iou_50', 'IoU ≥ 0.5', '#2a78d6'), ('iou_75', 'IoU ≥ 0.75', '#eb6834'), ('iou_90', 'IoU ≥ 0.9', '#1baf7a')]
# Masks -> GT outcome colours, as in the ShapBPT paper figure: white, navy, orange, magenta.
OUTCOME_NAMES = ['True positive', 'False negative', 'False positive']
OUTCOME_COLORS = ListedColormap(['#ffffff', '#1a1aa6', '#f28a30', '#f05cf0'])
SAMPLE_SIZE = 80  # Points drawn over each distribution, as in the notebook.
JUDGED = {'USE', 'CAUTION', 'SKIP'}


def style(ax, grid_axis='y'):
    ax.grid(axis=grid_axis, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRID)


def save(figure, folder, name):
    for suffix in ['png', 'svg']:
        figure.savefig(folder / f'{name}.{suffix}', dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def distribution(summaries, rows, column, ylabel, title, kind, folder, name):
    """Thesis Figures 4.8 and 4.9, with datasets in place of pipeline configurations."""
    labels = [item['config'] for item in summaries]
    data = [np.array([row[column] for row in rows if row['config'] == label]) for label in labels]
    figure, ax = plt.subplots(figsize=(max(8, 1.05 * len(labels)), 5.2))
    positions = np.arange(1, len(labels) + 1)
    if kind == 'box':
        ax.boxplot(data, positions=positions, showmeans=True, showfliers=False, widths=.55,
                   medianprops={'color': BASELINE, 'linewidth': 2}, boxprops={'color': MUTED},
                   whiskerprops={'color': MUTED}, capprops={'color': MUTED},
                   meanprops={'markerfacecolor': INK, 'markeredgecolor': 'white', 'marker': '^', 'markersize': 8})
    else:
        parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=False, widths=.8)
        for body in parts['bodies']:
            body.set(facecolor=PARTITION, edgecolor=PARTITION, alpha=.3)
        parts['cmeans'].set(color=INK, linewidth=1.5)
    rng = np.random.default_rng(42)
    for position, values in zip(positions, data):
        sample = rng.choice(values, size=min(SAMPLE_SIZE, len(values)), replace=False)
        ax.scatter(rng.normal(position, .04, len(sample)), sample, s=10, alpha=.35, color=PARTITION, edgecolors='none')
    ax.set_xticks(positions, labels, rotation=30, ha='right')
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, loc='left', fontweight='bold')
    style(ax)
    save(figure, folder, name)


def thresholds(summaries, folder):
    """Thesis Table 4.10 as a chart."""
    labels = [item['config'] for item in summaries]
    figure, ax = plt.subplots(figsize=(max(8, 1.05 * len(labels)), 4.6))
    positions = np.arange(len(labels))
    width = .26
    for index, (key, label, color) in enumerate(LEVELS):
        ax.bar(positions + (index - 1) * width, [item[key] * 100 for item in summaries], width * .9, color=color, label=label)
    ax.set_xticks(positions, labels, rotation=30, ha='right')
    ax.set_ylim(0, 100)
    ax.set_ylabel('GT objects (%)', color=MUTED)
    ax.set_title('Share of GT objects reaching each best IoU', color=INK, loc='left', fontweight='bold')
    ax.legend(frameon=False, labelcolor=MUTED, ncols=3, loc='upper right')
    style(ax)
    save(figure, folder, 'iou_thresholds')


def overview(summaries, args, folder):
    """One row per dataset: how far its partitions reach, against the grid and the verdict limits."""
    ordered = sorted(summaries, key=lambda item: item['median_iou'])
    figure, ax = plt.subplots(figsize=(9.5, max(3, .5 * len(ordered) + 1.6)))
    for row, item in enumerate(ordered):
        ax.plot([item['grid_median_iou'], item['median_iou']], [row, row], color=GRID, linewidth=2, zorder=1)
        ax.text(1.02, row, item['verdict'], va='center', color=INK, fontweight='bold', fontsize=9)
    rows = np.arange(len(ordered))
    ax.scatter([item['grid_median_iou'] for item in ordered], rows, s=60, marker='s', color=BASELINE,
               edgecolors='white', label='Same-size grid', zorder=2)
    ax.scatter([item['median_iou'] for item in ordered], rows, s=70, color=PARTITION, edgecolors='white',
               label='Partitions', zorder=3)
    ax.axvline(args.skip_median_iou, color=MUTED, linestyle=':', linewidth=1.2)
    ax.axvline(args.use_median_iou, color=MUTED, linestyle='--', linewidth=1.2)
    ax.text(args.skip_median_iou, len(ordered) - .35, 'SKIP below', ha='right', color=MUTED, fontsize=8)
    ax.text(args.use_median_iou, len(ordered) - .35, 'USE from', ha='left', color=MUTED, fontsize=8)
    ax.set_yticks(rows, [item['config'] for item in ordered])
    ax.set_xlim(0, 1)
    ax.set_ylim(-.6, len(ordered) - .1)
    ax.set_xlabel('Median best IoU per GT object', color=MUTED)
    ax.set_title('Can the partitions express the ground truth?', color=INK, loc='left', fontweight='bold', pad=12)
    ax.legend(frameon=False, labelcolor=MUTED, loc='lower right')
    style(ax, 'x')
    save(figure, folder, 'overview')


def failure_causes(summaries, diagnoses, folder):
    """What the objects below IoU 0.5 have in common, for each dataset."""
    ordered = sorted(summaries, key=lambda item: item['iou_50'])
    figure, ax = plt.subplots(figsize=(9.5, max(3, .5 * len(ordered) + 1.6)))
    rows = np.arange(len(ordered))
    left = np.zeros(len(ordered))
    for (name, _), color in zip(CAUSES.items(), ['#2a78d6', '#eb6834', '#1baf7a']):
        share = np.array([diagnoses[item['config']]['causes'][name] * (1 - item['iou_50']) * 100 for item in ordered])
        ax.barh(rows, share, left=left, height=.6, color=color, edgecolor='white', linewidth=2, label=name)
        left += share
    ax.set_yticks(rows, [item['config'] for item in ordered])
    ax.set_xlabel('Objects below IoU 0.5 (%)', color=MUTED)
    ax.set_title('Why objects score low', color=INK, loc='left', fontweight='bold', pad=26)
    ax.legend(frameon=False, labelcolor=MUTED, loc='lower left', bbox_to_anchor=(0, 1), ncols=3, borderaxespad=0)
    style(ax, 'x')
    save(figure, folder, 'failure_causes')


def iou_by_size(summaries, diagnoses, folder):
    """Median best IoU by object size, one panel per dataset on a shared scale."""
    columns = min(4, len(summaries))
    lines = -(-len(summaries) // columns)
    figure, axes = plt.subplots(lines, columns, figsize=(3.1 * columns, 2.9 * lines), sharey=True, squeeze=False)
    names = [name.split(' (')[0] for name, _ in SIZES]
    for ax in axes.ravel()[len(summaries):]:
        ax.set_visible(False)
    for ax, item in zip(axes.ravel(), summaries):
        found = {size['size'].split(' (')[0]: size for size in diagnoses[item['config']]['sizes']}
        ax.bar(names, [found[name]['median_iou'] if name in found else 0 for name in names], color=PARTITION, width=.6)
        for position, name in enumerate(names):
            if name in found:
                ax.text(position, found[name]['median_iou'] + .02, found[name]['objects'], ha='center', color=MUTED, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(item['config'], color=INK, loc='left', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        style(ax)
    figure.supylabel('Median best IoU', color=MUTED, fontsize=10)
    figure.suptitle('Best IoU by object size; numbers are object counts', color=INK, x=.01, ha='left', fontweight='bold')
    figure.tight_layout()
    save(figure, folder, 'iou_by_object_size')


def plot_all(summaries, rows, args, diagnoses):
    judged = [item for item in summaries if item['verdict'] in JUDGED]
    if not judged:
        print('No plots: no dataset has enough images for a verdict', flush=True)
        return
    folder = args.output_dir
    specs = [('best_iou', 'Best IoU', 'Best IoU by dataset'), ('gt_coverage', 'GT coverage (%)', 'GT coverage by dataset')]
    for column, ylabel, title in specs:
        distribution(judged, rows, column, ylabel, title, 'box', folder, f'{column}_boxplot')
        distribution(judged, rows, column, ylabel, title, 'violin', folder, f'{column}_violin')
    thresholds(judged, folder)
    overview(judged, args, folder)
    failure_causes(judged, diagnoses, folder)
    iou_by_size(judged, diagnoses, folder)


def outcome_map(truth, predicted):
    """Per pixel: 0 true negative, 1 true positive, 2 false negative, 3 false positive.

    A pixel given to the wrong object counts as a false negative, the object it belongs to being missed.
    """
    result = np.zeros(truth.shape, dtype=np.uint8)
    result[(predicted > 0) & (truth == 0)] = 3
    result[(truth > 0) & (predicted != truth)] = 2
    result[(truth > 0) & (predicted == truth)] = 1
    return result


def plot_examples(cases, label, folder):
    """Thesis Figures 4.10 to 4.12: one row each for the highest, middle, and lowest IoU image."""
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(cases), 5, figsize=(16, 3.4 * len(cases)), squeeze=False, constrained_layout=True)
    for row, (title, case) in zip(axes, cases.items()):
        objects = max(case['objects'], 1)
        values = np.linspace(.25, 1, objects)  # GT objects as gray levels on black, as in the notebook.
        colors = ListedColormap(np.vstack([[0, 0, 0, 1], np.column_stack([values, values, values, np.ones(objects)])]))
        labels = {'vmin': 0, 'vmax': objects, 'cmap': colors, 'interpolation': 'nearest'}
        row[0].imshow(case['image'])
        row[1].imshow(case['gt'], **labels)
        row[2].imshow(case['partition'], cmap='turbo', interpolation='nearest')
        row[3].imshow(case['subsets'], **labels)
        row[4].imshow(outcome_map(case['gt'], case['closest']), cmap=OUTCOME_COLORS, vmin=0, vmax=3,
                      interpolation='nearest')
        names = [title, f"Ground truth\n{case['objects']} object{'s' if case['objects'] != 1 else ''}", f"Partition\n{case['cells']} cells",
                 f"Optimal subsets\nMean IoU {case['mean_iou']:.3f}", f"Masks → GT\nMean IoU {case['closest_iou']:.3f}"]
        for ax, name in zip(row, names):
            ax.set_title(name, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
    figure.suptitle(label, fontsize=13)
    figure.legend(handles=[Patch(facecolor=color, edgecolor='#52514e', label=name)
                           for name, color in zip(OUTCOME_NAMES, OUTCOME_COLORS.colors[1:])],
                  loc='outside upper right', ncols=3, fontsize=10, frameon=False, title='Masks → GT')
    figure.savefig(folder / f"{label.replace(' ', '_')}.png", dpi=130, facecolor='white')
    plt.close(figure)
