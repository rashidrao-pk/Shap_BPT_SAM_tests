#!/usr/bin/env python3
"""Save a headless comparison of partition maps and their image overlays."""
import argparse
import copy
import json
import sys
from pathlib import Path
import yaml
from config_paths import output_path, partition_path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from partition_colors import colorize, count_title, region_count
import partition_files


def render(image_path, partition_paths, output, alpha=0.45, ground_truth=None, objects=None):
    """`objects`: (label map, names) of every GT object, drawn under the original instead of a second copy of it."""
    paths = [Path(path) for path in partition_paths]
    if not paths:
        raise ValueError('No partition maps selected')
    missing = [path for path in paths if not path.is_file()]
    if missing:
        folder, image_id = missing[0].parent, Path(image_path).stem
        present = sorted(path.name for path in folder.glob(f'{image_id}_*') if path.is_file())
        raise FileNotFoundError(f'{image_id}: not generated in {folder.name} yet (missing '
                                f'{", ".join(path.name for path in missing)}); files for this image there: '
                                f'{", ".join(present) or "none"}')
    image = np.asarray(Image.open(image_path).convert('RGB'))
    columns = len(paths) + 1 + int(ground_truth is not None)
    fig, axes = plt.subplots(2, columns, squeeze=False, figsize=(4 * columns, 7))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    if objects is not None and objects[1]:
        from partition_quality_plots import label_objects
        object_map, names = objects
        axes[1, 0].imshow(colorize(object_map), interpolation='nearest')  # Background black, one colour per object.
        label_objects(axes[1, 0], object_map, names)
        axes[1, 0].set_title(f"Ground truth\n{len(names)} object{'s' if len(names) != 1 else ''}")
    else:
        axes[1, 0].imshow(image)
        axes[1, 0].set_title('Original' if objects is None else 'Original (no GT objects)')
    statistics = []
    try:
        maps = []
        for path in paths:
            labels = np.load(path, allow_pickle=False)
            if labels.ndim != 2 or labels.shape != image.shape[:2] or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f'{path}: expected integer labels of shape {image.shape[:2]}')
            maps.append(labels)
        kinds = [stage_kind(path) for path in paths]
        sam_regions = region_count(maps[kinds.index('sam')]) if 'sam' in kinds else None
        for column, (path, labels, kind) in enumerate(zip(paths, maps, kinds), 1):
            unique, counts = np.unique(labels, return_counts=True)
            colored = colorize(labels)  # Label 0 (no SAM mask) is black.
            axes[0, column].imshow(colored)
            regions = region_count(labels)
            shown = kind_title(kind)
            title = (count_title(regions, sam_regions, shown) if kind != 'sam' and sam_regions is not None
                     else f'{shown.upper() if kind == "sam" else shown.capitalize()}\n{regions} regions')
            axes[0, column].set_title(f'{path.stem}\n{title}', fontsize=9)
            overlay = (1 - alpha) * (image / 255.) + alpha * colored
            axes[1, column].imshow(mark_boundaries(overlay, labels, color=(1, 1, 1)))
            axes[1, column].set_title('Overlay + boundaries')
            statistics.append({'path': str(path), 'shape': list(labels.shape),
                               'label_count': len(unique),
                               'zero_pixels': int(np.count_nonzero(labels == 0)),
                               'min_region_pixels': int(counts.min()),
                               'max_region_pixels': int(counts.max())})
        if ground_truth is not None:
            from ground_truth import title as gt_title, draw_ground_truth
            draw_ground_truth(axes[0, -1], ground_truth)
            axes[0, -1].set_title(gt_title(ground_truth), fontsize=9)
            draw_ground_truth(axes[1, -1], ground_truth, image=image, alpha=alpha, legend=True)
            axes[1, -1].set_title('GT overlay')
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)
        if ground_truth is not None:
            from ground_truth import save_ground_truth
            summary = save_ground_truth(ground_truth, image, output.parent, output.stem)
            statistics.append({'kind': 'ground_truth', **summary})
        output.with_suffix('.json').write_text(json.dumps(statistics, indent=2) + '\n')
    finally:
        plt.close(fig)
    return statistics


def stage_kind(path):
    """Stage of a partition file: 'sam', 'compact', 'compact_split', 'compact_mso_0.5', ..."""
    stem = Path(path).stem
    return stem[stem.rindex('_compact') + 1:] if '_compact' in stem else stem.rsplit('_', 1)[-1]


def kind_title(kind):
    """'compact_split' -> 'compact (split)'; the greedy and legacy files are plain 'compact'."""
    if not kind.startswith('compact') or partition_files.tag_of(kind) == partition_files.DEFAULT_TAG:
        return kind
    return f"compact ({partition_files.tag_of(kind).replace('_', ' ')})"


def method_stages(methods, data):
    """'compact' means the config's data.compact_strategy; other names are used as given."""
    return [partition_files.configured_stage(data) if method == 'compact' else method for method in methods]


def preview_ground_truth(args, image):
    if not args.show_gt:
        return None
    if not args.config:
        raise ValueError('--show_gt requires --config to locate dataset ground truth')
    from ground_truth import load_ground_truth
    _, config = load_view_config(args.config)
    with Image.open(image) as source:
        shape = (source.height, source.width, 3)
    return load_ground_truth(config, image, shape, gt_class=args.gt_class)


def preview_objects(args, image):
    """Every GT object of the image as (label map, names), largest painted first so small objects stay visible.

    None when the panel is turned off or the dataset has no usable GT; ([], []) when the image has no objects.
    """
    if not args.config or getattr(args, 'no_gt_panel', False):
        return None
    _, config = load_view_config(args.config)
    with Image.open(image) as source:
        shape = (source.height, source.width)
    try:
        from evaluate_partitions import gt_objects
        found = [(str(name), mask) for name, mask, _ in gt_objects(config, Path(image), shape)]
    except (KeyError, OSError, ValueError, TypeError) as error:  # No GT configured for this dataset.
        print(f'No ground-truth panel for {Path(image).stem}: {type(error).__name__}: {error}', file=sys.stderr)
        return None
    found.sort(key=lambda item: -int(item[1].sum()))
    object_map = np.zeros(shape, dtype=np.int64)
    for number, (_, mask) in enumerate(found, 1):
        object_map[mask] = number
    present = [number for number in range(1, len(found) + 1) if (object_map == number).any()]
    relabel = np.zeros(len(found) + 1, dtype=np.int64)
    relabel[present] = np.arange(1, len(present) + 1)
    return relabel[object_map], [found[number - 1][0] for number in present]


def load_view_config(config_name, sam=None, partition_folder=None):
    """Load a config; `sam` ('sam1'/'sam2') or `partition_folder` replaces data.mask_dir_final."""
    root = Path(__file__).resolve().parents[2]
    path = Path(config_name).expanduser()
    if not path.suffix:
        path = root / 'src/configs' / f'{config_name}.yaml'
    elif not path.is_absolute():
        path = root / path
    config = yaml.safe_load(path.read_text())
    data = config['data']
    if partition_folder:
        data['mask_dir_final'] = partition_folder
    elif sam:
        key = f'mask_dir_final_{sam}'
        if not data.get(key):
            raise ValueError(f'{path.name} has no data.{key}; pass --partition-folder FOLDER_NAME instead')
        data['mask_dir_final'] = data[key]
    return root, config


def partition_choice(args):
    return {'sam': getattr(args, 'sam', None), 'partition_folder': getattr(args, 'partition_folder', None)}


def folder_hint(folder, data):
    """Which folders of this dataset exist, and the --sam / --partition-folder option that selects each."""
    import re
    prefix = re.split(r'_(?:sam1|sam2|medsam|medicosam)_', folder.name)[0]
    existing = sorted(path.name for path in folder.parent.glob(f'{prefix}_*') if path.is_dir()) \
        if folder.parent.is_dir() else []
    if not existing:
        return f'. No {prefix}_* partition folders exist yet; generate them first'
    by_key = {value: key.removeprefix('mask_dir_final_') for key, value in data.items()
              if key.startswith('mask_dir_final_') and isinstance(value, str)}
    options = [f'--sam {by_key[name]}' if name in by_key else f'--partition-folder {name}' for name in existing]
    return '. Existing folders: ' + '; '.join(f'{name} ({option})' for name, option in zip(existing, options))


def discover_image_ids(config_name, methods, limit, allowed_ids=None, **choice):
    root, config = load_view_config(config_name, **choice)
    data = config['data']
    folder = partition_path(config, root)
    if not folder.is_dir():
        raise ValueError(f'Partition folder not found: {folder}{folder_hint(folder, data)}')
    image_ids = []
    for metadata in sorted(folder.glob('*_results.json')):
        image_id = metadata.name.removesuffix('_results.json')
        if allowed_ids is not None and image_id not in allowed_ids:
            continue
        if all(partition_files.partition_file(folder, image_id, method).is_file()
               for method in method_stages(methods, data)):
            image_ids.append(image_id)
            if len(image_ids) >= limit:
                break
    if not image_ids:
        names = {path.name for path in folder.iterdir() if path.is_file()}
        counts = ', '.join(f'{stage}: {len(partition_files.stage_files(folder, stage, names))}'
                           for stage in method_stages(methods, data))
        raise ValueError(f'No completed images with all requested stages in {folder} '
                         f'(images per stage: {counts})')
    return image_ids


def partimagenet_image_index(config):
    from partimagenet_data import collect_images
    from config_paths import resolve_path
    data = config['data']
    image_dir = resolve_path(data['image_dir'])
    return {item['id']: image_dir / item['file_name'] for item in collect_images(data)}


def resolve_paths(args, image_index=None):
    """Resolve config paths; explicit CLI paths take precedence."""
    image, partitions, output = args.image, args.partitions, args.output
    if args.config:
        root, config = load_view_config(args.config, **partition_choice(args))
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

        partition_dir = partition_path(config, root)
        if partitions is None:
            partitions = [partition_files.partition_file(partition_dir, image_id, method)
                          for method in method_stages(args.methods, data)]
        if image is None:
            image_dir = rooted(data['image_dir'])
            if data.get('name', '').lower() in {'partimagenet', 'paco'}:
                images = image_index if image_index is not None else partimagenet_image_index(config)
                if image_id not in images:
                    raise ValueError(f'{image_id}: not listed in the selected annotation split')
                image = images[image_id]
            else:
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
            output_root = output_path(config, 'evaluations', root)
            output = output_root / 'previews' / data['mask_dir_final'] / f'{image_id}.png'
    if image is None or partitions is None or output is None:
        raise ValueError('Use --config with --image-id, or provide --image, --partitions, and --output')
    return image, partitions, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Config name or YAML path')
    parser.add_argument('--image-id', '--image-ids', dest='image_ids', nargs='+', metavar='ID',
                        help='One or more COCO numeric IDs or filename stems')
    parser.add_argument('--limit', type=int,
                        help='Preview first N completed images from the config partition folder')
    parser.add_argument('--methods', nargs='+', default=['sam', 'refined'],
                        help='Partition stages to view (default: sam refined). compact = the config\'s '
                             'data.compact_strategy; name others directly, e.g. sam compact_greedy compact_split')
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--sam', choices=['sam1', 'sam2', 'medsam', 'medicosam'],
                        help="Partitions of this model: the config's data.mask_dir_final_<model> (sam1, sam2, medsam, medicosam) "
                             '(default: data.mask_dir_final)')
    choice.add_argument('--partition-folder', help='Partition folder name under output.partitions, '
                                                   'e.g. MSCOCO_sam1_vit_b_62_compact-sam_200')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--partitions', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--alpha', type=float, default=0.45)
    parser.add_argument('--show_gt', '--show-gt', action='store_true', help='Include dataset ground truth in the preview')
    parser.add_argument('--gt-class', help='Optional face part, ImageNetS synset/label ID, or foreground')
    parser.add_argument('--no-gt-panel', action='store_true',
                        help='Show the original twice instead of the GT objects in the lower-left panel')
    args = parser.parse_args()
    args.image_id = args.image_ids[0] if args.image_ids else None
    if args.image_ids and len(args.image_ids) > 1 and any([args.image, args.partitions, args.output]):
        parser.error('several --image-id values cannot be combined with --image, --partitions or --output')
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1')
    if args.limit is not None:
        if args.limit < 1 or not args.config:
            parser.error('--limit must be positive and requires --config')
        if any([args.image_id, args.image, args.partitions, args.output]):
            parser.error('--limit cannot be combined with explicit image/partition/output paths')
    try:
        if args.limit is not None:
            _, config = load_view_config(args.config, **partition_choice(args))
            image_index = (partimagenet_image_index(config)
                           if config['data'].get('name', '').lower() in {'partimagenet', 'paco'} else None)
            for image_id in discover_image_ids(args.config, args.methods, args.limit, image_index, **partition_choice(args)):
                selected = copy.copy(args)
                selected.image_id = image_id
                image, partitions, output = resolve_paths(selected, image_index)
                render(image, partitions, output, args.alpha, preview_ground_truth(args, image), preview_objects(args, image))
                print(f'Preview: {output}')
            return
        if args.image_ids and len(args.image_ids) > 1:
            failed = []
            for image_id in args.image_ids:
                selected = copy.copy(args)
                selected.image_id = image_id
                try:
                    image, partitions, output = resolve_paths(selected)
                    render(image, partitions, output, args.alpha, preview_ground_truth(args, image), preview_objects(args, image))
                except (OSError, ValueError) as exc:  # One missing image must not stop the others.
                    failed.append(image_id)
                    print(f'{image_id}: {exc}', file=sys.stderr)
                    continue
                print(f'Preview: {output}')
            if failed:
                parser.error(f'no preview for {len(failed)} of {len(args.image_ids)} images: {" ".join(failed)}')
            return
        image, partitions, output = resolve_paths(args)
        statistics = render(image, partitions, output, args.alpha, preview_ground_truth(args, image),
                            preview_objects(args, image))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(statistics, indent=2))
    print(f'Preview: {output}')


if __name__ == '__main__':
    main()
