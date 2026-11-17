#!/usr/bin/env python3
"""BPT build time with 1, 2 and 4 label words (T1 step 6).

    python src/scripts/analysis/bpt_build_time.py --config MSCOCO_epito --limit 20 --out results/paper/T1
    python src/scripts/analysis/bpt_build_time.py --images-dir DIR --partition-dir DIR --limit 20 --out ...

For each image and W, the compact partition is capped to 64 * W - 1 labels (as the runners do), the guided
tree is built `--repeats` times, and the median time is kept. An unguided build is timed as the reference
(W = 0). Writes build_time.csv (image, W, n_labels, seconds) and a README with the command.
"""
import argparse
from pathlib import Path
import platform
import shlex
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/scripts')]


def inputs(args):
    if args.config:
        from config_paths import load_path_config, partition_path
        from evaluate_partitions import image_paths
        config = load_path_config(args.config)
        images, folder = image_paths(config['data']), partition_path(config)
    else:
        folder = Path(args.partition_dir)
        images = {path.stem: path for path in Path(args.images_dir).glob('*.jpg')}
    ids = sorted(set(path.name.removesuffix('_compact.npy') for path in folder.glob('*_compact.npy')) & set(images))
    if not ids:
        raise SystemExit(f'No images with a compact partition in {folder}')
    order = np.random.default_rng(args.seed).permutation(len(ids))[:args.limit]
    return [(ids[index], images[ids[index]], folder) for index in sorted(order)]


def timed(build, repeats):
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        build()
        seconds.append(time.perf_counter() - started)
    return float(np.median(seconds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', help='Dataset config; its data.mask_dir_final is used')
    parser.add_argument('--images-dir', help='Instead of --config: folder of <id>.jpg')
    parser.add_argument('--partition-dir', help='Instead of --config: folder of <id>_compact.npy')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--words', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/paper/T1')
    args = parser.parse_args(argv)
    if not args.config and not (args.images_dir and args.partition_dir):
        parser.error('give --config, or both --images-dir and --partition-dir')
    import pandas as pd
    from PIL import Image
    import shap_bpt
    from utils_sam_2 import load_refine_partitions
    rows = []
    for image_id, image_path, folder in inputs(args):
        with Image.open(image_path) as source:
            image = np.asarray(source.convert('RGB'))
        rows.append({'image': image_id, 'W': 0, 'n_labels': 0, 'height': image.shape[0], 'width': image.shape[1],
                     'seconds': timed(lambda: shap_bpt.build_bpt_from_image(image), args.repeats)})
        for words in args.words:
            _, labels = load_refine_partitions(image, str(folder), image_id, 'compact', max_labels=64 * words - 1)
            rows.append({'image': image_id, 'W': words, 'n_labels': int(len(np.unique(labels))),
                         'height': image.shape[0], 'width': image.shape[1],
                         'seconds': timed(lambda: shap_bpt.build_bpt_from_image(
                             image, prebuilt_partitions=labels, label_words=words), args.repeats)})
        print(f"{image_id}: " + ', '.join(f"W={row['W']} {row['seconds']:.3f}s ({row['n_labels']} labels)"
                                          for row in rows if row['image'] == image_id), flush=True)
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / 'build_time.csv', index=False)
    summary = table.groupby('W').agg(images=('image', 'nunique'), median_labels=('n_labels', 'median'),
                                     median_seconds=('seconds', 'median'), mean_seconds=('seconds', 'mean'))
    command = 'python src/scripts/analysis/bpt_build_time.py ' + ' '.join(shlex.quote(item) for item in (argv or sys.argv[1:]))
    (args.out / 'README.md').write_text(
        '# BPT build time by label words (T1)\n\n' f'```bash\n{command}\n```\n\n'
        f'Machine: {platform.platform()}, {platform.processor() or platform.machine()}, Python {platform.python_version()}.\n\n'
        '`build_time.csv`: one row per image and W. W = 0 is the unguided BPT; W = 1, 2, 4 are guided builds with the '
        'partition capped to 64W - 1 labels. `seconds` is the median of the repeats.\n\n'
        + summary.round(4).to_markdown() + '\n')
    print(summary.round(4).to_string(), flush=True)
    return table


if __name__ == '__main__':
    main()
