#!/usr/bin/env python3
"""Save a headless comparison of partition maps and their image overlays."""
import argparse
import copy
import json
from pathlib import Path
import yaml

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


def render(image_path, partition_paths, output, alpha=0.45):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    paths = [Path(path) for path in partition_paths]
    if not paths:
        raise ValueError('No partition maps selected')
    fig, axes = plt.subplots(2, len(paths) + 1, squeeze=False,
                             figsize=(4 * (len(paths) + 1), 7))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original')
    statistics = []
    try:
        for column, path in enumerate(paths, 1):
            labels = np.load(path, allow_pickle=False)
            if labels.ndim != 2 or labels.shape != image.shape[:2] or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f'{path}: expected integer labels of shape {image.shape[:2]}')
            unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
            colors = plt.colormaps['turbo'](np.linspace(0.05, 0.95, len(unique)))[:, :3]
            colored = colors[inverse.reshape(labels.shape)]
            axes[0, column].imshow(colored)
            axes[0, column].set_title(f'{path.stem}\n{len(unique)} labels', fontsize=9)
            overlay = (1 - alpha) * (image / 255.) + alpha * colored
            axes[1, column].imshow(mark_boundaries(overlay, labels, color=(1, 1, 1)))
            axes[1, column].set_title('Overlay + boundaries')
            statistics.append({'path': str(path), 'shape': list(labels.shape),
                               'label_count': len(unique),
                               'zero_pixels': int(np.count_nonzero(labels == 0)),
                               'min_region_pixels': int(counts.min()),
                               'max_region_pixels': int(counts.max())})
        for ax in axes.ravel():
            ax.axis('off')
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)
        output.with_suffix('.json').write_text(json.dumps(statistics, indent=2) + '\n')
    finally:
        plt.close(fig)
    return statistics


def load_view_config(config_name):
    root = Path(__file__).resolve().parents[2]
    path = Path(config_name).expanduser()
    if not path.suffix:
        path = root / 'src/configs' / f'{config_name}.yaml'
    elif not path.is_absolute():
        path = root / path
    return root, yaml.safe_load(path.read_text())


def discover_image_ids(config_name, methods, limit):
    root, config = load_view_config(config_name)
    data = config['data']
    base = Path(data['masks_path']).expanduser()
    if not base.is_absolute():
        base = root / base
    folder = base / data['mask_dir'] / data['mask_dir_final']
    if not folder.is_dir():
        raise ValueError(f'Partition folder not found: {folder}; check data.mask_dir_final')
    image_ids = []
    for metadata in sorted(folder.glob('*_results.json')):
        image_id = metadata.name.removesuffix('_results.json')
        if all((folder / f'{image_id}_{method}.npy').is_file() for method in methods):
            image_ids.append(image_id)
            if len(image_ids) >= limit:
                break
    if not image_ids:
        raise ValueError(f'No completed images with all requested stages in {folder}')
    return image_ids


def resolve_paths(args):
    """Resolve config paths; explicit CLI paths take precedence."""
    image, partitions, output = args.image, args.partitions, args.output
    if args.config:
        root, config = load_view_config(args.config)
        data = config['data']
        image_id = args.image_id or (image.stem if image else None)
        if image_id is None:
            raise ValueError('--config requires --image-id or --image')
        if data.get('input_format') != 'directory' and image_id.isdigit():
            image_id = f'{int(image_id):012d}'
        elif data.get('name', '').lower() == 'imagenet' and image_id.isdigit():
            image_id = f'ILSVRC2012_val_{int(image_id):08d}'

        def rooted(value):
            path = Path(value).expanduser()
            return path if path.is_absolute() else root / path

        partition_dir = rooted(data['masks_path']) / data['mask_dir'] / data['mask_dir_final']
        if partitions is None:
            partitions = [partition_dir / f'{image_id}_{method}.npy' for method in args.methods]
        if image is None:
            image_dir = rooted(data['image_dir'])
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
            candidates = [p for p in image_dir.glob(f'{image_id}.*')
                          if p.suffix.lower() in extensions and p.is_file()]
            if not candidates and data.get('input_format') == 'directory':
                candidates = [p for p in image_dir.rglob(f'{image_id}.*')
                              if p.suffix.lower() in extensions and p.is_file()]
            if len(candidates) != 1:
                raise ValueError(f'Expected one image for {image_id} under {image_dir}; '
                                 f'found {len(candidates)}. Pass --image explicitly.')
            image = candidates[0]
        if output is None:
            settings = config.get('output', {})
            if not settings.get('dir') and not settings.get('folder'):
                raise ValueError('Config needs output.dir/output.folder, or pass --output')
            output_root = rooted(settings.get('dir', '.')) / settings.get('folder', '')
            output = output_root / 'previews' / data['mask_dir_final'] / f'{image_id}.png'
    if image is None or partitions is None or output is None:
        raise ValueError('Use --config with --image-id, or provide --image, --partitions, and --output')
    return image, partitions, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Config name or YAML path')
    parser.add_argument('--image-id', help='COCO numeric ID or ImageNet filename stem')
    parser.add_argument('--limit', type=int,
                        help='Preview first N completed images from the config partition folder')
    parser.add_argument('--methods', nargs='+', default=['sam', 'refined'],
                        help='Partition suffixes to view (default: sam refined)')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--partitions', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--alpha', type=float, default=0.45)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1')
    if args.limit is not None:
        if args.limit < 1 or not args.config:
            parser.error('--limit must be positive and requires --config')
        if any([args.image_id, args.image, args.partitions, args.output]):
            parser.error('--limit cannot be combined with explicit image/partition/output paths')
    try:
        if args.limit is not None:
            for image_id in discover_image_ids(args.config, args.methods, args.limit):
                selected = copy.copy(args)
                selected.image_id = image_id
                image, partitions, output = resolve_paths(selected)
                render(image, partitions, output, args.alpha)
                print(f'Preview: {output}')
            return
        image, partitions, output = resolve_paths(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(render(image, partitions, output, args.alpha), indent=2))
    print(f'Preview: {output}')


if __name__ == '__main__':
    main()
