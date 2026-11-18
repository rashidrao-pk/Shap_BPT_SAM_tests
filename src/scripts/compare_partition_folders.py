#!/usr/bin/env python3
"""Compare compact strategies (files in one partition folder) or partition folders, image by image.

For each image it saves one figure: the original and GT, the raw SAM map, and each strategy's or folder's
partition with its region count and GT score. It also writes per-image and summary tables. Scores follow evaluate_partitions.py:
for every GT object, the best IoU reachable by a union of cells, and the same for a plain grid with as many cells.
No model or GPU is needed.

    # Strategies saved in one folder (<id>_compact_greedy.npy, <id>_compact_split.npy, ...):
    python src/scripts/compare_partition_folders.py --config MSCOCO_epito \
      --folders MSCOCO_sam2_large_254_compact-sam_200 --partition-types compact_greedy compact_mso_0.5 compact_split \
      --image-id 76211 28285 57725 113867 11615
    # Folders (e.g. SAM-1 vs SAM-2), each with the config's compact strategy:
    python src/scripts/compare_partition_folders.py --config MSCOCO_epito \
      --folders MSCOCO_sam1_vit_b_254_compact-sam_200 MSCOCO_sam2_large_254_compact-sam_200
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

ROOT = Path(__file__).resolve().parents[2]
for directory in [ROOT, ROOT / 'src/scripts']:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import evaluate_partitions as evaluation  # noqa: E402
from config_paths import output_path  # noqa: E402
from partition_colors import colorize, count_title, region_count  # noqa: E402
import partition_files  # noqa: E402
from view_partitions import load_view_config, preview_objects  # noqa: E402

PER_IMAGE_FIELDS = ['image_id', 'label', 'folder', 'partition_type', 'sam_regions', 'regions', 'unassigned_pct', 'objects',
                    'mean_iou', 'median_iou', 'iou_50_pct', 'grid_mean_iou', 'gain']
SUMMARY_FIELDS = ['label', 'folder', 'partition_type', 'images', 'mean_regions', 'mean_unassigned_pct', 'objects', 'mean_iou',
                  'median_iou', 'iou_50_pct', 'grid_mean_iou', 'gain', 'best_on_images']


def folder_label(folder):
    """Short name from the generator's folder suffix: `..._greedy` -> Greedy, `..._mso_0.5` -> MSO α=0.5."""
    found = re.search(r'_mso_([0-9.eE+-]+)$', folder)
    if found:
        return f'MSO α={found.group(1)}'
    if folder.endswith('_greedy'):
        return 'Greedy'
    found = re.search(r'_split(?:_(\d+))?$', folder)
    if found:
        return f'Split (gain ≥{found.group(1)} px)' if found.group(1) else 'Split'
    return folder


def type_label(partition_type):
    """Short name of a compact file: compact_greedy -> Greedy, compact_mso_0.5 -> MSO α=0.5, compact_split -> Split."""
    if not partition_type.startswith('compact'):
        return partition_type
    strategy, alpha, gain = partition_files.parse_tag(partition_files.tag_of(partition_type))
    if strategy == 'mso':
        return f'MSO α={alpha:g}'
    if strategy == 'split':
        return 'Split' if gain == partition_files.DEFAULT_MIN_GAIN else f'Split (gain ≥{gain} px)'
    return 'Greedy'


def entries_and_labels(folders, types, labels=None):
    """(folder, partition type) pairs to compare and their display names."""
    if len(folders) == 1:
        entries = [(folders[0], kind) for kind in types]
    elif len(types) == 1:
        entries = [(folder, types[0]) for folder in folders]
    elif len(types) == len(folders):
        entries = list(zip(folders, types))
    else:
        raise SystemExit('Give one folder and several --partition-types, several folders and one type, '
                         'or one type per folder')
    if labels:
        if len(labels) != len(entries):
            raise SystemExit(f'--labels needs {len(entries)} names')
        return entries, list(labels)
    multi_folder, multi_type = len({f for f, _ in entries}) > 1, len({t for _, t in entries}) > 1
    names = [' '.join(part for part in [folder_label(folder.name) if multi_folder else '',
                                        type_label(kind) if multi_type or not multi_folder else ''] if part)
             for folder, kind in entries]
    if len(set(names)) != len(names):
        names = [f'{folder.name}:{kind}' for folder, kind in entries]
    return entries, names


def id_keys(value):
    text = str(value).strip()
    return {text, str(int(text))} if text.isdigit() else {text}


def available_ids(folder, partition_type):
    return set(partition_files.stage_files(folder, partition_type))


def select_ids(entries, partition_type=None, wanted=None, limit=None):
    """Image IDs present for every (folder, type) entry; `wanted` may use padded or unpadded numbers.
    Plain folders with one `partition_type` are accepted too."""
    entries = [(entry, partition_type) if partition_type else entry for entry in entries]
    common = set.intersection(*(available_ids(folder, kind) for folder, kind in entries))
    if wanted:
        chosen, missing = [], []
        for requested in wanted:
            keys = id_keys(requested)
            match = sorted(image_id for image_id in common if id_keys(image_id) & keys)
            (chosen.extend(match) if match else missing.append(str(requested)))
        if missing:
            print(f'Not in every folder (generate them first): {", ".join(missing)}', file=sys.stderr)
        result = list(dict.fromkeys(chosen))
    else:
        result = sorted(common)
    return result[:limit] if limit else result


def score(config, label, image_id, image_path, partition_file, combine):
    rows = evaluation.evaluate_image(config, label, image_id, image_path, partition_file, combine)
    partition = evaluation.load_partition(image_path, partition_file)
    sam_file = partition_file.with_name(f'{image_id}_sam.npy')  # One SAM map per folder and image
    ious = np.array([row['best_iou'] for row in rows])
    grid = np.array([row['grid_iou'] for row in rows])
    return {
        'image_id': image_id, 'label': label, 'folder': partition_file.parent.name,
        'partition_type': partition_file.name[len(image_id) + 1:-4],
        'sam_regions': region_count(evaluation.load_partition(image_path, sam_file)) if sam_file.is_file() else '',
        'regions': region_count(partition), 'unassigned_pct': float((partition == 0).mean() * 100),
        'objects': len(rows),
        'mean_iou': float(ious.mean()) if len(rows) else np.nan,
        'median_iou': float(np.median(ious)) if len(rows) else np.nan,
        'iou_50_pct': float((ious >= .5).mean() * 100) if len(rows) else np.nan,
        'grid_mean_iou': float(grid.mean()) if len(rows) else np.nan,
        'gain': float(ious.mean() - grid.mean()) if len(rows) else np.nan,
        '_ious': ious, '_grid': grid, '_partition': partition,
    }


def draw(image_path, image_id, results, objects, output, alpha=0.45):
    """Row 1: original, raw SAM, each folder's map. Row 2: GT objects, SAM overlay, each folder's overlay."""
    image = np.asarray(Image.open(image_path).convert('RGB'))
    sam_file = Path(results[0]['_file']).with_name(f'{image_id}_sam.npy')
    sam = evaluation.load_partition(image_path, sam_file) if sam_file.is_file() else None
    columns = 1 + (sam is not None) + len(results)
    fig, axes = plt.subplots(2, columns, squeeze=False, figsize=(3.6 * columns, 6.8))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Original\n{image_id}', fontsize=9)
    if objects is not None and objects[1]:
        from partition_quality_plots import label_objects
        axes[1, 0].imshow(colorize(objects[0]), interpolation='nearest')
        label_objects(axes[1, 0], *objects)
        axes[1, 0].set_title(f'Ground truth\n{len(objects[1])} objects', fontsize=9)
    else:
        axes[1, 0].imshow(image)
        axes[1, 0].set_title('No GT objects', fontsize=9)
    panels = ([('SAM (raw masks)', sam, None)] if sam is not None else []) + [
        (row['label'], row['_partition'], row) for row in results]
    best = max((row['mean_iou'] for row in results if not np.isnan(row['mean_iou'])), default=None)
    for column, (name, labels, row) in enumerate(panels, 1):
        colored = colorize(labels)
        axes[0, column].imshow(colored, interpolation='nearest')
        if row is None:
            title = f'{name}\n{region_count(labels)} regions'
        else:
            title = f"{name}\n{count_title(row['regions'], row['sam_regions'] or None, 'compact').replace(chr(10), ' | ')}"
            if row['objects']:
                star = ' ★' if best is not None and row['mean_iou'] == best and len(results) > 1 else ''
                title += f"\nmean IoU {row['mean_iou']:.2f} (grid {row['grid_mean_iou']:.2f}){star}"
        axes[0, column].set_title(title, fontsize=9)
        overlay = (1 - alpha) * (image / 255.) + alpha * colored
        axes[1, column].imshow(mark_boundaries(overlay, labels, color=(1, 1, 1)))
        if row is not None:
            axes[1, column].set_title(f"unassigned {row['unassigned_pct']:.1f}% | IoU≥0.5 {row['iou_50_pct']:.0f}%"
                                      if row['objects'] else f"unassigned {row['unassigned_pct']:.1f}%", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def summarize(per_image, labels):
    summary = []
    winners = {}
    for image_id in dict.fromkeys(row['image_id'] for row in per_image):
        rows = [row for row in per_image if row['image_id'] == image_id and row['objects']]
        if rows:
            top = max(row['mean_iou'] for row in rows)
            for row in rows:
                if row['mean_iou'] == top:
                    winners[row['label']] = winners.get(row['label'], 0) + 1
    for label in labels:
        rows = [row for row in per_image if row['label'] == label]
        ious = np.concatenate([row['_ious'] for row in rows]) if rows else np.array([])
        grid = np.concatenate([row['_grid'] for row in rows]) if rows else np.array([])
        has = len(ious) > 0
        summary.append({
            'label': label, 'folder': rows[0]['folder'] if rows else '',
            'partition_type': rows[0]['partition_type'] if rows else '', 'images': len(rows),
            'mean_regions': float(np.mean([row['regions'] for row in rows])) if rows else np.nan,
            'mean_unassigned_pct': float(np.mean([row['unassigned_pct'] for row in rows])) if rows else np.nan,
            'objects': len(ious), 'mean_iou': float(ious.mean()) if has else np.nan,
            'median_iou': float(np.median(ious)) if has else np.nan,
            'iou_50_pct': float((ious >= .5).mean() * 100) if has else np.nan,
            'grid_mean_iou': float(grid.mean()) if has else np.nan,
            'gain': float(ious.mean() - grid.mean()) if has else np.nan,
            'best_on_images': winners.get(label, 0)})
    return summary


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f'{value:.4f}' if isinstance(value, float) else value)
                             for key, value in row.items() if key in fields})


def print_summary(summary):
    header = f"{'Setting':<16}{'images':>7}{'regions':>9}{'unassigned%':>13}{'objects':>9}" \
             f"{'mean IoU':>10}{'median':>8}{'IoU≥.5%':>9}{'grid':>7}{'gain':>7}{'best on':>9}"
    print(header)
    print('-' * len(header))
    for row in summary:
        print(f"{row['label']:<16}{row['images']:>7}{row['mean_regions']:>9.1f}{row['mean_unassigned_pct']:>13.1f}"
              f"{row['objects']:>9}{row['mean_iou']:>10.3f}{row['median_iou']:>8.3f}{row['iou_50_pct']:>9.1f}"
              f"{row['grid_mean_iou']:>7.3f}{row['gain']:>7.3f}{row['best_on_images']:>9}")


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--config', required=True, help='Config name or YAML path (for images, GT and output.partitions)')
    result.add_argument('--folders', nargs='+', help='Partition folder names under output.partitions '
                        '(default: the config\'s data.mask_dir_final)')
    result.add_argument('--partition-types', '--partition-type', dest='partition_types', nargs='+',
                        help='Stage files to compare, e.g. compact_greedy compact_split '
                             '(default: the config\'s data.compact_strategy)')
    result.add_argument('--labels', nargs='+', help='Display names, one per compared entry')
    result.add_argument('--image-id', '--image-ids', dest='image_ids', nargs='+', metavar='ID',
                        help='Images to compare (default: every image present in all folders, up to --limit)')
    result.add_argument('--limit', type=int, help='Most images to compare')
    result.add_argument('--name', default='compare', help='Output subfolder name')
    result.add_argument('--output-dir', type=Path,
                        help='Default: <output.evaluations>/partition_compare/<name>')
    result.add_argument('--no-figures', dest='figures', action='store_false')
    result.add_argument('--no-gt-panel', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    root, config = load_view_config(args.config)
    base = output_path(config, 'partitions', root)
    folders = [base / name for name in (args.folders or [config['data']['mask_dir_final']])]
    missing = [str(folder) for folder in folders if not folder.is_dir()]
    if missing:
        raise SystemExit('Partition folder(s) not found:\n  ' + '\n  '.join(missing))
    types = [partition_files.stage(kind) if kind.startswith('compact') else kind
             for kind in (args.partition_types or [partition_files.configured_stage(config['data'])])]
    entries, labels = entries_and_labels(folders, types, args.labels)
    image_ids = select_ids(entries, None, args.image_ids, args.limit)
    if not image_ids:
        raise SystemExit('No image has every compared partition')
    paths = evaluation.image_paths(config['data'])
    combine = config['data'].get('name', '').lower() == 'celebahq'
    output = args.output_dir or output_path(config, 'evaluations', root) / 'partition_compare' / args.name
    output.mkdir(parents=True, exist_ok=True)

    per_image = []
    for number, image_id in enumerate(image_ids, 1):
        image_path = paths.get(image_id)
        if image_path is None:
            print(f'{image_id}: image not found in data.image_dir, skipped', file=sys.stderr)
            continue
        rows = []
        for (folder, kind), label in zip(entries, labels):
            partition_file = partition_files.partition_file(folder, image_id, kind)
            row = score(config, label, image_id, image_path, partition_file, combine)
            row['_file'] = partition_file
            rows.append(row)
        per_image.extend(rows)
        line = ' | '.join(f"{row['label']}: {row['regions']} regions, IoU {row['mean_iou']:.2f}" for row in rows)
        print(f'[{number}/{len(image_ids)}] {image_id}: {line}', flush=True)
        if args.figures:
            objects = None if args.no_gt_panel else preview_objects(
                SimpleNamespace(config=args.config, no_gt_panel=False), image_path)
            draw(image_path, image_id, rows, objects, output / f'{image_id}.png')

    summary = summarize(per_image, labels)
    write_csv(output / 'per_image.csv', per_image, PER_IMAGE_FIELDS)
    write_csv(output / 'summary.csv', summary, SUMMARY_FIELDS)
    print()
    print_summary(summary)
    print(f'\nFigures and tables: {output}')
    return summary


if __name__ == '__main__':
    main()
