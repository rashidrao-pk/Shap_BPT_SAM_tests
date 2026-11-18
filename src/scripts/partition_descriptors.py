#!/usr/bin/env python3
"""GT-free partition descriptors, one row per image (T-0d). No model, no GPU, no ground truth.

    python src/scripts/partition_descriptors.py --all-sam --paper-dir results/paper/T-0d

Per image, from the saved compact map and its *_results.json:
  sam_masks          raw SAM masks after mask-count tuning
  regions_raw        non-zero labels in the saved compact map
  regions_after_cap  non-zero labels the explainers see after the 255-label cap
  target_met         whether the mask-count target was reached
  unassigned_share   share of pixels in label 0 (no SAM mask)
  generation_sec     recorded per-image generation time
  lab_variance       within-region colour variance: mean over pixels of the squared CIELAB distance to their
                     region's mean colour (label 0 counts as one region). Lower = more homogeneous regions.
  boundary_recall    share of Canny edge pixels (grey image, sigma 2, hysteresis at the 75th/90th percentile of
                     gradient magnitude) within 2 px of a partition boundary.
                     Higher = the partition follows the image's edges.
Both scores describe the partition alone; neither says whether the regions match objects.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ['dataset', 'segmenter', 'partitions', 'image_id', 'sam_masks', 'regions_raw', 'regions_after_cap',
          'target_met', 'unassigned_share', 'generation_sec', 'lab_variance', 'boundary_recall']
SUMMARY = ['regions_after_cap', 'unassigned_share', 'generation_sec', 'lab_variance', 'boundary_recall']


def lab_variance(image_rgb, labels):
    """Pixel-weighted within-region variance in CIELAB (sum over L, a, b)."""
    from skimage.color import rgb2lab
    with np.errstate(all='ignore'):  # Spurious matmul warnings from some BLAS builds; values are checked below.
        lab = rgb2lab(image_rgb).reshape(-1, 3)
    if not np.isfinite(lab).all():
        raise ValueError('CIELAB conversion produced non-finite values')
    flat = np.unique(labels.ravel(), return_inverse=True)[1]
    counts = np.bincount(flat).astype(float)
    means = np.stack([np.bincount(flat, weights=lab[:, channel]) / counts for channel in range(3)], axis=1)
    return float(((lab - means[flat]) ** 2).sum(axis=1).mean())


def boundary_recall(image_rgb, labels, tolerance=2, sigma=2.0):
    """Share of Canny edge pixels within `tolerance` pixels of a partition boundary; NaN without edges."""
    from scipy import ndimage as ndi
    from skimage.color import rgb2gray
    from skimage.feature import canny
    # Quantile thresholds (75th/90th percentile of gradient magnitude) keep edge density comparable across
    # images of different contrast; absolute thresholds miss low-contrast colour edges.
    edges = canny(rgb2gray(image_rgb), sigma=sigma, low_threshold=.75, high_threshold=.9, use_quantiles=True)
    if not edges.any():
        return float('nan')
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
    near = ndi.distance_transform_edt(~boundary) <= tolerance if boundary.any() else np.zeros_like(boundary)
    return float((edges & near).sum() / edges.sum())


def describe(image_rgb, partition, metadata):
    from utils_sam_2 import cap_partition_labels, sanitize_partitions
    capped = sanitize_partitions(cap_partition_labels(partition, max_labels=255), image_rgb.shape, max_labels=255)
    seconds = metadata.get('total_time_sec')
    return {'sam_masks': metadata.get('n_sam_masks'), 'regions_raw': int(np.count_nonzero(np.unique(partition))),
            'regions_after_cap': int(np.count_nonzero(np.unique(capped))), 'target_met': metadata.get('mask_target_met'),
            'unassigned_share': float((partition == 0).mean()),
            'generation_sec': float(seconds) if isinstance(seconds, (int, float)) else None,
            'lab_variance': lab_variance(image_rgb, capped), 'boundary_recall': boundary_recall(image_rgb, capped)}


def folder_rows(name, folder, args):
    from PIL import Image
    from check_partition_progress import sam_label
    from config_paths import load_path_config
    from evaluate_partitions import image_paths, load_partition
    config = load_path_config(name)
    paths = image_paths(config['data'])
    from partition_files import configured_stage, stage_files
    files = stage_files(folder, args.partition_type or configured_stage(config['data']))
    ids = sorted(set(files) & set(paths))
    if args.limit:
        ids = [ids[index] for index in sorted(np.random.default_rng(args.seed).permutation(len(ids))[:args.limit])]
    base = {'dataset': Path(name).stem.removesuffix('_epito'), 'segmenter': sam_label(folder.name), 'partitions': folder.name}
    rows = []
    for number, image_id in enumerate(ids, 1):
        try:
            with Image.open(paths[image_id]) as source:
                image = np.asarray(source.convert('RGB'))
            partition = load_partition(paths[image_id], files[image_id])
            meta_path = folder / f'{image_id}_results.json'
            metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            rows.append({**base, 'image_id': image_id, **describe(image, partition, metadata)})
        except (OSError, ValueError) as error:  # One unreadable image must not stop the folder.
            print(f'{base["dataset"]} {image_id}: skipped ({type(error).__name__}: {error})', flush=True)
        if number % 200 == 0 or number == len(ids):
            print(f'{base["dataset"]} {base["segmenter"]}: {number}/{len(ids)}', flush=True)
    return rows


def summarize(rows):
    import pandas as pd
    table = pd.DataFrame(rows, columns=FIELDS)
    if table.empty:
        return table
    grouped = table.groupby(['dataset', 'segmenter', 'partitions'], sort=False)
    summary = grouped[SUMMARY].median().add_suffix('_median')
    summary.insert(0, 'images', grouped.size())
    summary['target_met_share'] = grouped['target_met'].apply(
        lambda values: float(values.dropna().astype(bool).mean()) if values.notna().any() else np.nan)
    return summary.reset_index()


def main(argv=None):
    sys.path.insert(0, str(ROOT / 'src/scripts'))
    from check_partition_progress import DEFAULT_CONFIGS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', nargs='+', default=DEFAULT_CONFIGS)
    parser.add_argument('--all-sam', action='store_true', help="Every SAM folder of each dataset, not only the config's")
    parser.add_argument('--partition-type', help='Stage file, e.g. compact_split (default: data.compact_strategy)')
    parser.add_argument('--limit', type=int, help='Random images per folder (default: all)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/evaluations/partition_descriptors')
    parser.add_argument('--paper-dir', type=Path, help='Also write the tables and a README here, e.g. results/paper/T-0d')
    args = parser.parse_args(argv)
    from config_paths import load_path_config
    from evaluate_partitions import partition_folders
    rows, errors = [], []
    for name in args.config:
        try:
            for folder in partition_folders(load_path_config(name), args.all_sam):
                rows.extend(folder_rows(name, folder, args))
        except (OSError, ValueError, KeyError) as error:
            errors.append(f'{name}: {type(error).__name__}: {error}')
            print(f'ERROR {errors[-1]}', flush=True)
    import pandas as pd
    summary = summarize(rows)
    for folder in [args.output_dir, args.paper_dir]:
        if folder is None:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=FIELDS).to_csv(folder / 'partition_descriptors.csv', index=False)
        summary.to_csv(folder / 'partition_descriptors_summary.csv', index=False)
    if args.paper_dir:
        command = 'python src/scripts/partition_descriptors.py ' + ' '.join(shlex.quote(item) for item in (argv or sys.argv[1:]))
        (args.paper_dir / 'README.md').write_text(
            '# GT-free partition descriptors (T-0d)\n\n' f'```bash\n{command}\n```\n\n'
            '`partition_descriptors.csv`: one row per image and partition folder. '
            '`partition_descriptors_summary.csv`: medians per folder and the share of images that met the mask target. '
            'Definitions are in the docstring of `src/scripts/partition_descriptors.py`.'
            + (f'\n\nErrors: {"; ".join(errors)}\n' if errors else '\n'))
    if not summary.empty:
        print(summary.round(3).to_string(index=False), flush=True)
    print(f'Saved: {args.output_dir}' + (f' and {args.paper_dir}' if args.paper_dir else ''), flush=True)
    return summary


if __name__ == '__main__':
    main()
