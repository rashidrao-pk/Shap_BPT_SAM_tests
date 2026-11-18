#!/usr/bin/env python3
"""Animate actual BPT and SAM-guided BPT merges for saved dataset partitions (CPU only)."""
import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src/scripts')]


def region_schedule(pixels, start, frames):
    """Fine to coarse, with more visible stages near the root; end at one region."""
    count = min(pixels, start)
    return sorted({int(value) for value in np.rint(np.geomspace(1, count, frames))}, reverse=True)


def tree_cut(tree, count):
    """Undo the newest merges to obtain exactly count active nodes of the actual tree."""
    if not 1 <= count <= tree.U:
        raise ValueError('Cut count outside tree leaf range')
    nodes = [-(int(tree.N) - 1)]
    while len(nodes) < count:
        node = -heapq.heappop(nodes)
        offset = node - int(tree.U)
        heapq.heappush(nodes, -int(tree.cl_left[offset]))
        heapq.heappush(nodes, -int(tree.cl_right[offset]))
    labels = np.empty((tree.height, tree.width), dtype=np.int32)
    anchors = []
    for label, negative in enumerate(sorted(nodes)):
        node = -negative
        if node < tree.U:
            start = int(tree.leaf_idx[node])
            end = start + 1
        else:
            start, end = int(tree.cl_start[node-tree.U]), int(tree.cl_end[node-tree.U])
        pixels = np.asarray(tree.pixels[start:end], dtype=int)
        labels.ravel()[pixels] = label
        anchors.append(int(pixels.min()))
    return labels, np.asarray(anchors)


def colored_cut(image, labels, anchors=None):
    """Stable colours from a region's anchor pixel, with boundaries on the input image."""
    from skimage.segmentation import find_boundaries
    _, inverse = np.unique(labels, return_inverse=True)
    indices = inverse.reshape(labels.shape)
    if anchors is None:
        anchors = np.full(indices.max() + 1, indices.size, dtype=np.int64)
        np.minimum.at(anchors, inverse.ravel(), np.arange(indices.size))
    # Deterministic colour, independent of Python hash randomisation.
    colors = np.stack([((anchors * factor + shift) % 193 + 40) / 255.
                       for factor, shift in [(37, 31), (67, 79), (97, 127)]], axis=-1)
    result = .35 * image.astype(float) / 255 + .65 * colors[indices]
    result[find_boundaries(indices, mode='inner')] = 0
    return np.uint8(np.clip(result * 255, 0, 255))


def render(image, guidance, trees, output, title, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    schedule = region_schedule(image.shape[0] * image.shape[1], args.start_regions, args.frames)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.3))
    axes[0].imshow(image)
    axes[0].set_title('Input')
    axes[1].imshow(colored_cut(image, guidance))
    axes[1].set_title(f'SAM guidance: {len(np.unique(guidance))} labels')
    artists = [ax.imshow(image) for ax in axes[2:]]
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    heading = fig.suptitle(title)
    fig.tight_layout(rect=(0, .04, 1, .90))
    fig.text(.5, .015, 'Fine → coarse | matching region counts | intermediate merges omitted', ha='center', fontsize=9)

    def update(frame):
        count = schedule[frame]
        for ax, artist, tree, name in zip(axes[2:], artists, trees, ['BPT', 'SAM-guided BPT']):
            labels, anchors = tree_cut(tree, count)
            artist.set_data(colored_cut(image, labels, anchors))
            ax.set_title(f'{name}\n{count} regions')
        heading.set_text(f'{title}\nStage {frame + 1}/{len(schedule)} · {count} regions')
        return artists

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        update(0)
        fig.savefig(output.with_suffix('.png'), dpi=args.dpi)
        animation = FuncAnimation(fig, update, frames=len(schedule), interval=1000 / args.fps, blit=False)
        animation.save(output, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    finally:
        plt.close(fig)
    return schedule


def animate_one(config_name, config, image_id, image_path, partition_folder, args):
    import shap_bpt
    from explanations import guide_options
    from partition_files import configured_stage, partition_file
    from utils_sam_2 import load_refine_partitions, sanitize_partitions

    stage = args.partition_type or configured_stage(config['data'])
    with Image.open(image_path) as source:
        image = np.asarray(source.convert('RGB'))
    original_shape = image.shape[:2]
    _, guidance = load_refine_partitions(image, str(partition_folder), image_id, stage,
                                         max_labels=guide_options.max_labels(args))
    # Full-resolution is the default. Optional resize changes the trees and is recorded.
    if args.max_side and max(image.shape[:2]) > args.max_side:
        scale = args.max_side / max(image.shape[:2])
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
        guidance = np.asarray(Image.fromarray(guidance.astype(np.int32)).resize(size, Image.Resampling.NEAREST))
        guidance = sanitize_partitions(guidance, image.shape, max_labels=guide_options.max_labels(args))
    times, trees = [], []
    for name, kwargs in [('BPT', {}), ('SAM-guided BPT', dict(prebuilt_partitions=guidance,
                                                           **guide_options.tree_kwargs(args)))]:
        print(f'{config_name} {image_id}: building {name} at {image.shape[1]}×{image.shape[0]}', flush=True)
        started = time.perf_counter()
        trees.append(shap_bpt.build_bpt_from_image(image, **kwargs))
        times.append(time.perf_counter() - started)
    from config_paths import output_path
    output_root = args.output_dir or output_path(config, 'evaluations') / 'bpt_animations'
    variant = stage + guide_options.folder_suffix(args) + (f'_maxside{args.max_side}' if args.max_side else '')
    output = output_root / Path(config_name).stem / partition_folder.name / variant / f'{image_id}.gif'
    schedule = render(image, guidance, trees, output,
                      f"{config['data']['name']} · {image_id} · {stage}", args)
    metadata = {'config': str(config_name), 'image_id': image_id, 'image': str(image_path),
                'partition_file': str(partition_file(partition_folder, image_id, stage)),
                'original_shape': original_shape, 'animated_shape': image.shape[:2],
                'guide_settings': guide_options.tree_kwargs(args), 'guidance_labels': len(np.unique(guidance)),
                'region_counts': schedule, 'fps': args.fps, 'build_seconds': dict(zip(['BPT', 'SAM-guided BPT'], times)),
                'note': 'Actual pixel-leaf BPT trees. Equal-count cuts in merge order; intermediate merges omitted.'}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved: {output}', flush=True)
    return str(output)


def parser():
    from check_partition_progress import DEFAULT_CONFIGS
    from explanations import guide_options
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', nargs='+', default=DEFAULT_CONFIGS)
    result.add_argument('--limit', type=int, default=1, help='Images per dataset, default 1')
    result.add_argument('--image-id', nargs='+', help='Specific image stems; COCO IDs may be unpadded')
    result.add_argument('--sam', choices=['sam1', 'sam2'], help='Read mask_dir_final_sam1 or mask_dir_final_sam2')
    result.add_argument('--partition-folder', help='Override partition folder (single config only)')
    result.add_argument('--partition-type', help='Stage, default compact strategy from config')
    result.add_argument('--start-regions', type=int, default=256)
    result.add_argument('--frames', type=int, default=48)
    result.add_argument('--fps', type=float, default=5)
    result.add_argument('--dpi', type=int, default=100)
    result.add_argument('--max-side', type=int, default=0, help='Optional preview resize; 0 keeps original resolution')
    result.add_argument('--output-dir', type=Path, help='Default: output.evaluations/bpt_animations')
    guide_options.add_arguments(result)
    return result


def main(argv=None):
    from config_paths import load_path_config, output_path, partition_path
    from evaluate_partitions import image_paths
    from partition_files import configured_stage, stage_files
    args = parser().parse_args(argv)
    if min(args.limit, args.start_regions, args.fps, args.dpi) <= 0 or args.frames < 2 or args.max_side < 0:
        raise SystemExit('Limits, regions, fps and dpi must be positive; frames >= 2; max-side >= 0')
    if args.partition_folder and len(args.config) != 1:
        raise SystemExit('--partition-folder requires one config')
    failures, outputs = [], []
    for name in args.config:
        try:
            config = load_path_config(name)
            if args.sam:
                config['data']['mask_dir_final'] = config['data'][f'mask_dir_final_{args.sam}']
            folder = (output_path(config, 'partitions') / args.partition_folder
                      if args.partition_folder else partition_path(config))
            stage = args.partition_type or configured_stage(config['data'])
            files = stage_files(folder, stage)
            if not files:
                raise FileNotFoundError(f'No {stage} partitions in {folder}')
            paths = image_paths(config['data'])
            ids = sorted(set(paths) & set(files))
            if args.image_id:
                keys = lambda text: str(int(text)) if str(text).isdigit() else str(text)
                wanted = {keys(value) for value in args.image_id}
                ids = [value for value in ids if keys(value) in wanted]
                missing = wanted - {keys(value) for value in ids}
                if missing:
                    raise ValueError('Requested images missing input/partition: ' + ', '.join(sorted(missing)))
            if not ids:
                raise ValueError('No source images match completed partitions')
            for image_id in ids[:args.limit]:
                try:
                    outputs.append(animate_one(name, config, image_id, paths[image_id], folder, args))
                except Exception as error:
                    failures.append(f'{name} {image_id}: {type(error).__name__}: {error}')
                    print(f'FAILED {failures[-1]}', flush=True)
        except Exception as error:
            failures.append(f'{name}: {type(error).__name__}: {error}')
            print(f'FAILED {failures[-1]}', flush=True)
    print(f'Animations completed: {len(outputs)}; failures: {len(failures)}', flush=True)
    if failures:
        raise SystemExit(1)
    return outputs


if __name__ == '__main__':
    main()
