#!/usr/bin/env python3
"""Check whether each dataset's partitions can express its ground truth, before running XAI.

For every GT object or part, this finds the union of partition cells with the highest IoU, as in
notebooks/stats.ipynb, and compares it with a plain grid that has the same number of cells.
It needs no model or GPU. It writes tables, plots, and a verdict for each dataset, which cover
partition quality only, not model behaviour.
"""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for directory in [ROOT, ROOT / 'src/scripts']:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from check_partition_progress import DEFAULT_CONFIGS
from config_paths import load_path_config, partition_path, resolve_path
from ground_truth import _coco, annotation_mask, load_ground_truth
from partition_diagnosis import diagnose, failure_cause, image_table, report, size_band
from utils_sam_2 import load_partition_array

IGNORED_LABEL = 1000  # ImageNet-S marks pixels to ignore with this value.
IOU_LEVELS = (.5, .75, .9)
USE, CAUTION, SKIP = 'USE', 'CAUTION', 'SKIP'
NEED_PARTITIONS, NO_GT, ERROR = 'NEED PARTITIONS', 'NO GT', 'ERROR'
NUMBERS = ['found', 'images', 'failed', 'objects', 'cells', 'mean_iou', 'std_iou', 'median_iou', 'grid_median_iou', 'gain',
           'iou_50', 'iou_75', 'iou_90', 'mean_coverage', 'std_coverage', 'median_coverage']
SUMMARY_FIELDS = ['config', 'partitions', *NUMBERS, 'verdict', 'reason']
OBJECT_FIELDS = ['config', 'image_id', 'object', 'gt_pixels', 'gt_share', 'size', 'cells', 'best_iou', 'grid_iou',
                 'gt_coverage', 'precision', 'unassigned', 'cause']
IMAGE_FIELDS = ['config', 'image_id', 'mean_iou', 'objects', 'low_objects', 'cells', 'smallest_object', 'main_cause']


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--config', nargs='+', default=DEFAULT_CONFIGS)
    result.add_argument('--limit', type=int, help='Images with GT to evaluate per dataset; default: every completed partition')
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--partition-type', default='compact')
    result.add_argument('--all-sam', action='store_true', help="Also evaluate the dataset's other SAM partition folders")
    result.add_argument('--min-images', type=int, default=30, help='Fewer images than this gives no verdict')
    result.add_argument('--use-median-iou', type=float, default=.75)
    result.add_argument('--use-iou-50', type=float, default=.75, help='Share of GT objects with IoU of at least 0.5')
    result.add_argument('--skip-median-iou', type=float, default=.5)
    result.add_argument('--min-gain', type=float, default=.05, help='Median IoU lead over the same-size grid')
    result.add_argument('--output-dir', type=Path, default=ROOT / 'results/evaluations/partition_quality')
    result.add_argument('--no-plots', dest='plots', action='store_false')
    result.add_argument('--worst', type=int, default=6, help='Worst images to list and draw per dataset')
    result.add_argument('--split-parts', action='store_true', help='Score each CelebA part alone, not the combined face')
    return result


def maximize_iou(gt_mask, partition):
    """Best IoU of any union of cells, its GT coverage in percent, and those cells; cell 0 is background.

    Adding cells in order of purity (share of the cell inside the GT) reaches the optimum.
    """
    gt_mask = np.asarray(gt_mask, dtype=bool).ravel()
    cells = np.asarray(partition, dtype=np.int64).ravel()
    gt_area = int(gt_mask.sum())
    inside = np.bincount(cells, weights=gt_mask)
    inside[0] = 0
    ids = np.nonzero(inside)[0]
    if gt_area == 0 or ids.size == 0:
        return 0., 0., []
    total = np.bincount(cells)[ids]
    overlap = inside[ids]
    order = np.argsort(-overlap / total, kind='stable')
    covered = np.cumsum(overlap[order])
    outside = np.cumsum((total - overlap)[order])
    ious = covered / (gt_area + outside)
    best = int(np.argmax(ious))
    return float(ious[best]), float(covered[best] / gt_area * 100), ids[order[:best + 1]].tolist()


def grid_partition(shape, cells):
    """A near-square grid of at least the given number of cells, numbered from 1."""
    height, width = shape
    rows = min(height, max(1, round(np.sqrt(cells * height / width))))
    columns = min(width, -(-cells // rows))
    y = np.arange(height) * rows // height
    x = np.arange(width) * columns // width
    return y[:, None] * columns + x[None, :] + 1


def image_paths(data):
    name = data.get('name', '').replace('_', '').lower()
    folder = resolve_path(data['image_dir'])
    if name == 'mscoco':
        return {path.stem: path for path in folder.glob('*.jpg')}
    if name in {'partimagenet', 'paco'}:
        from partimagenet_data import collect_images
        return {item['id']: folder / item['file_name'] for item in collect_images(data)}
    if name == 'busi':
        from src.medical.datasets import discover_images
        return {item['id']: folder / item['file_name'] for item in discover_images(folder)}
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    return {path.stem: path for path in folder.rglob('*') if path.suffix.lower() in extensions}


def combines_parts(config, args):
    """CelebA parts are scored as one face, because most single parts are far smaller than any cell."""
    return config['data'].get('name', '').lower() == 'celebahq' and not args.split_parts


def gt_objects(config, image_path, shape, combine=False):
    """Yield (name, mask, valid pixels) for each GT object or part, or once for all of them."""
    data = config['data']
    if data.get('name', '').replace('_', '').lower() == 'mscoco':
        coco = _coco(str(resolve_path(data['annotation_file'])))
        names = {item['id']: item['name'] for item in coco.loadCats(coco.getCatIds())}
        for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[int(image_path.stem)], iscrowd=False)):
            mask = annotation_mask(coco, annotation)
            if mask is not None:
                yield names[annotation['category_id']], mask, None
        return
    if (data.get('ground_truth') or {}).get('format', '').lower() == 'dota':
        yield from dota_objects(config, image_path, shape)
        return
    gt = load_ground_truth(config, image_path, shape)
    labels = np.asarray(gt['label_map'])
    names = gt.get('label_names') or {}
    valid = labels != IGNORED_LABEL
    if combine:
        if gt['mask'].any():
            yield 'face', gt['mask'], valid
        return
    for value in np.unique(labels[gt['mask']]):
        yield names.get(int(value), gt.get('dataset_class', str(value))), gt['mask'] & (labels == value), valid


def dota_objects(config, image_path, shape):
    """One full mask per counted DOTA object, so overlapping boxes do not cut each other.

    Ignored regions (difficult or border-cut objects) are left out of the scored pixels,
    except where they overlap a counted object.
    """
    from dota_data import tile_instances
    root = resolve_path(config['data']['ground_truth']['root'])
    path = root / f'{Path(image_path).stem}.txt'
    if not path.is_file():
        return
    instances = tile_instances(path, shape)
    counted = [item for item in instances if not item['difficult']]
    ignored = np.zeros(shape[:2], dtype=bool)
    for item in instances:
        if item['difficult']:
            ignored |= item['mask']
    for item in counted:
        ignored &= ~item['mask']
    valid = ~ignored if ignored.any() else None
    for item in counted:
        if item['mask'].any():
            yield item['class_name'], item['mask'], valid


def load_partition(image_path, partition_file):
    with Image.open(image_path) as source:
        shape = (source.height, source.width)
    partition = load_partition_array(partition_file, shape).astype(np.int64)
    if partition.shape != shape:
        raise ValueError(f'{Path(image_path).stem}: partition {partition.shape} does not match image {shape}')
    return partition


def evaluate_image(config, name, image_id, image_path, partition_file, combine=False):
    partition = load_partition(image_path, partition_file)
    shape = partition.shape
    cells = int(np.count_nonzero(np.unique(partition)))
    grid = grid_partition(shape, max(cells, 1))
    rows = []
    for object_name, mask, valid in gt_objects(config, image_path, shape, combine):
        selected = np.ones(shape, dtype=bool) if valid is None else valid
        iou, coverage, ids = maximize_iou(mask[selected], partition[selected])
        grid_iou = maximize_iou(mask[selected], grid[selected])[0]
        chosen = np.isin(partition[selected], ids)
        precision = float((chosen & mask[selected]).sum() / max(chosen.sum(), 1))
        unassigned = float((partition[mask] == 0).mean()) if mask.any() else 0.
        share = float(mask.mean() * 100)
        rows.append({'config': name, 'image_id': image_id, 'object': object_name, 'gt_pixels': int(mask.sum()), 'gt_share': share,
                     'size': size_band(share), 'cells': cells, 'best_iou': iou, 'grid_iou': grid_iou, 'gt_coverage': coverage,
                     'precision': precision, 'unassigned': unassigned, 'cause': failure_cause(iou, precision, unassigned)})
    return rows


def case_maps(config, image_path, partition_file, combine=False):
    """Label maps for one example: GT, the best union of cells, and each cell's closest GT object."""
    partition = load_partition(image_path, partition_file)
    with Image.open(image_path) as source:
        image = np.asarray(source.convert('RGB'))
    gt_labels, subsets = np.zeros_like(partition), np.zeros_like(partition)
    cells = int(partition.max()) + 1
    area = np.bincount(partition.ravel(), minlength=cells)
    closest, owner = np.zeros(cells), np.zeros(cells, dtype=np.int64)
    ious = []
    for number, (_, mask, valid) in enumerate(gt_objects(config, Path(image_path), partition.shape, combine), 1):
        selected = np.ones(partition.shape, dtype=bool) if valid is None else valid
        iou, _, ids = maximize_iou(mask[selected], partition[selected])
        ious.append(iou)
        gt_labels[mask] = number
        subsets[np.isin(partition, ids)] = number
        overlap = np.bincount(partition[mask], minlength=cells)
        cell_iou = overlap / np.maximum(area + mask.sum() - overlap, 1)
        better = cell_iou > closest
        closest[better], owner[better] = cell_iou[better], number
    owner[0] = 0
    closest_map = owner[partition]
    return {'image': image, 'partition': partition, 'gt': gt_labels, 'subsets': subsets, 'closest': closest_map,
            'objects': len(ious), 'cells': int(np.count_nonzero(np.unique(partition))), 'mean_iou': float(np.mean(ious)),
            'closest_iou': label_iou(gt_labels, closest_map, len(ious))}


def label_iou(truth, predicted, objects):
    """Mean over GT objects of the IoU between the pixels labelled with that object in each map."""
    scores = []
    for number in range(1, objects + 1):
        union = np.count_nonzero((truth == number) | (predicted == number))
        scores.append(np.count_nonzero((truth == number) & (predicted == number)) / union if union else 0.)
    return float(np.mean(scores)) if scores else 0.



def example_ids(rows):
    """Images with the highest, middle, and lowest mean best IoU, as in the thesis examples."""
    means = {}
    for row in rows:
        means.setdefault(row['image_id'], []).append(row['best_iou'])
    ordered = sorted(means, key=lambda image_id: (np.mean(means[image_id]), image_id))
    return {'Highest IoU': ordered[-1], 'Middle IoU': ordered[len(ordered) // 2], 'Lowest IoU': ordered[0]}


def decide(summary, args):
    """Return the verdict and its reason. Default limits sit just below the COCO SAM-1 reference."""
    if summary['found'] < args.min_images:
        return NEED_PARTITIONS, f"only {summary['found']} partitions; need {args.min_images}"
    if summary['images'] < args.min_images:
        return NO_GT, f"only {summary['images']} images with GT; need {args.min_images}"
    median, gain, share = summary['median_iou'], summary['gain'], summary['iou_50']
    if median < args.skip_median_iou:
        return SKIP, f'median IoU {median:.2f} is below {args.skip_median_iou:g}'
    if gain <= 0:
        return SKIP, f'no better than a plain grid (gain {gain:+.2f})'
    problems = []
    if median < args.use_median_iou:
        problems.append(f'median IoU {median:.2f} is below {args.use_median_iou:g}')
    if share < args.use_iou_50:
        problems.append(f'only {share:.0%} of objects reach IoU 0.5')
    if gain < args.min_gain:
        problems.append(f'small gain over grid ({gain:+.2f})')
    return (CAUTION, '; '.join(problems)) if problems else (USE, f'median IoU {median:.2f}, {share:.0%} reach IoU 0.5')


def summarize(name, folder, found, images, rows, args, failed=0):
    summary = {**dict.fromkeys(NUMBERS, 0.), 'config': name, 'partitions': folder, 'found': found, 'images': images,
               'objects': len(rows), 'failed': failed}
    if rows:
        ious = np.array([row['best_iou'] for row in rows])
        grids = np.array([row['grid_iou'] for row in rows])
        coverage = np.array([row['gt_coverage'] for row in rows])
        per_image = {row['image_id']: row['cells'] for row in rows}
        summary.update(cells=float(np.median(list(per_image.values()))), mean_iou=float(ious.mean()),
                       std_iou=float(ious.std(ddof=1)) if len(rows) > 1 else 0., median_iou=float(np.median(ious)),
                       grid_median_iou=float(np.median(grids)), gain=float(np.median(ious - grids)),
                       mean_coverage=float(coverage.mean()), std_coverage=float(coverage.std(ddof=1)) if len(rows) > 1 else 0.,
                       median_coverage=float(np.median(coverage)),
                       **{f'iou_{round(level * 100)}': float((ious >= level).mean()) for level in IOU_LEVELS})
    summary['verdict'], summary['reason'] = decide(summary, args)
    if failed:
        summary['reason'] += f'; {failed} images could not be read'
    return summary


def partition_folders(config, all_sam):
    folder = partition_path(config)
    if not all_sam:
        return [folder]
    prefix = folder.name.split('_sam')[0]
    return sorted({folder, *(path for path in folder.parent.glob(f'{prefix}_sam*') if path.is_dir())})


def evaluate_folder(name, config, folder, args):
    suffix = f'_{args.partition_type}.npy'
    files = {path.name.removesuffix(suffix): path for path in folder.glob(f'*{suffix}')} if folder.is_dir() else {}
    paths = image_paths(config['data']) if files else {}
    ids = sorted(set(files) & set(paths))
    label = Path(name).stem.removesuffix('_epito')
    label = f"{label} {folder.name.split('_')[1].upper()}" if args.all_sam else label
    combine = combines_parts(config, args)
    rows, images, failed = [], 0, 0
    for index in np.random.default_rng(args.seed).permutation(len(ids)):
        if images == args.limit:
            break
        try:
            image_rows = evaluate_image(config, label, ids[index], paths[ids[index]], files[ids[index]], combine)
        except (OSError, TypeError, ValueError) as error:  # One bad image must not lose the whole dataset.
            failed += 1
            print(f'{label} {ids[index]}: {type(error).__name__}: {error}', flush=True)
            continue
        rows.extend(image_rows)
        images += bool(image_rows)  # Images without GT do not use up the limit.
    summary = summarize(label, folder.name, len(ids), images, rows, args, failed)
    if rows and args.plots and summary['verdict'] in {USE, CAUTION, SKIP}:
        from partition_quality_plots import plot_examples
        cases = {title: case_maps(config, paths[image_id], files[image_id], combine) for title, image_id in example_ids(rows).items()}
        plot_examples(cases, label, args.output_dir / 'examples')
        worst = {f"Worst {number}\n{item['image_id']}": case_maps(config, paths[item['image_id']], files[item['image_id']], combine)
                 for number, item in enumerate(image_table(rows)[:args.worst], 1)}
        plot_examples(worst, label, args.output_dir / 'worst')
    return summary, rows


def evaluate_dataset(name, args):
    config = load_path_config(name)
    return [evaluate_folder(name, config, folder, args) for folder in partition_folders(config, args.all_sam)]


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def conclusion(summaries, args):
    lines = ['# Partition quality', '', 'Best reachable IoU between partition cells and each GT object. '
             'This shows whether the partitions can express the GT; it does not show that a model uses those regions.', '',
             f'USE needs a median IoU of at least {args.use_median_iou:g}, at least {args.use_iou_50:.0%} of objects at IoU 0.5, '
             f'and a median lead over a same-size grid of at least {args.min_gain:g}. '
             f'SKIP means a median IoU below {args.skip_median_iou:g}, or no lead over the grid.', '',
             '| Dataset | Verdict | Images | Objects | Median IoU | Grid | IoU ≥ 0.5 | IoU ≥ 0.75 | IoU ≥ 0.9 | Median coverage | Why |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for item in summaries:
        lines.append(f"| {item['config']} | **{item['verdict']}** | {item['images']} | {item['objects']} | {item['median_iou']:.3f} | "
                     f"{item['grid_median_iou']:.3f} | {item['iou_50']:.1%} | {item['iou_75']:.1%} | {item['iou_90']:.1%} | "
                     f"{item['median_coverage']:.1f}% | {item['reason']} |")
    return '\n'.join(lines) + '\n'


def print_table(summaries):
    print(f"\n{'Dataset':<22} {'Verdict':<16} {'Images':>6} {'Objects':>8} {'Cells':>6} {'Mean':>6} {'SD':>6} {'Median':>7} {'Grid':>6} "
          f"{'>=.5':>6} {'>=.75':>6} {'>=.9':>6} {'Cover':>6}")
    for item in summaries:
        print(f"{item['config']:<22} {item['verdict']:<16} {item['images']:>6} {item['objects']:>8} {item['cells']:>6.0f} "
              f"{item['mean_iou']:>6.3f} {item['std_iou']:>6.3f} {item['median_iou']:>7.3f} {item['grid_median_iou']:>6.3f} "
              f"{item['iou_50']:>6.1%} {item['iou_75']:>6.1%} {item['iou_90']:>6.1%} {item['median_coverage']:>5.1f}%")
    print()
    for item in summaries:
        print(f"{item['config']}: {item['verdict']} - {item['reason']}")


def run(args):
    if (args.limit is not None and args.limit < 1) or args.min_images < 1 or args.worst < 1:
        raise ValueError('--limit, --min-images, and --worst must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, rows = [], []
    for name in args.config:
        print(f'Evaluating {name}', flush=True)
        try:
            for summary, dataset_rows in evaluate_dataset(name, args):
                summaries.append(summary)
                rows.extend(dataset_rows)
        except Exception as error:  # One broken dataset should not hide the others.
            summaries.append({**dict.fromkeys(NUMBERS, 0), 'config': Path(name).stem.removesuffix('_epito'), 'partitions': '',
                              'verdict': ERROR, 'reason': f'{type(error).__name__}: {error}'})
    write_csv(args.output_dir / 'partition_quality_objects.csv', rows, OBJECT_FIELDS)
    diagnoses = {item['config']: diagnose([row for row in rows if row['config'] == item['config']])
                 for item in summaries if item['verdict'] in {USE, CAUTION, SKIP}}
    write_csv(args.output_dir / 'partition_quality_images.csv',
              [image for result in diagnoses.values() for image in result['images']], IMAGE_FIELDS)
    (args.output_dir / 'diagnosis.md').write_text(report(summaries, diagnoses, args.worst))
    write_csv(args.output_dir / 'partition_quality_summary.csv', summaries, SUMMARY_FIELDS)
    (args.output_dir / 'conclusion.md').write_text(conclusion(summaries, args))
    if args.plots:
        from partition_quality_plots import plot_all
        plot_all(summaries, rows, args, diagnoses)
    print_table(summaries)
    print(f'\nSaved: {args.output_dir}', flush=True)
    return summaries


if __name__ == '__main__':
    run(parser().parse_args())
