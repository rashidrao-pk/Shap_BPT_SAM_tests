#!/usr/bin/env python3
"""Run notebook greedy selection, compact labels, booster, filling, and previews."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

import sam_notebook_pipeline as pipeline
from view_partitions import render


def process(image_path, masks, scores, output_dir, select_count=400,
            max_distance=3, create_gif=False, save_step_images=False):
    image_path, output_dir = Path(image_path), Path(output_dir)
    with Image.open(image_path) as source:
        shape = (source.height, source.width)
    masks, scores = np.asarray(masks, dtype=bool), np.asarray(scores, dtype=float)
    if masks.ndim != 3 or masks.shape[1:] != shape:
        raise ValueError(f'Expected masks [N,{shape[0]},{shape[1]}], got {masks.shape}')
    if scores.shape != (len(masks),) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Expected one finite, nonnegative predicted_iou score per mask')
    if select_count < 1:
        raise ValueError('select_count must be positive')
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    annotations = [{'segmentation': mask, 'area': int(mask.sum()), 'predicted_iou': float(score)}
                   for mask, score in zip(masks, scores)]
    selected, _, history, gif = pipeline.select_masks_by_unpicked_area(
        annotations, select_count, create_gif=create_gif,
        gif_path=str(output_dir / f'{image_path.stem}_selection.gif'),
        save_step_images=save_step_images,
        step_images_dir=str(output_dir / f'{image_path.stem}_steps'))
    zero = np.zeros(shape, dtype=np.int32)
    raw = pipeline.create_compact_mask(annotations) if annotations else zero.copy()
    compact = pipeline.create_compact_mask(selected) if selected else zero.copy()
    boosted = pipeline.refiner(compact, selected) if selected else zero.copy()
    filled = pipeline.partition_unassigned_by_nearest_cluster(compact, max_distance=max_distance)
    boosted_filled = pipeline.partition_unassigned_by_nearest_cluster(boosted, max_distance=max_distance)
    maps = {'sam': raw, 'compact_selection': compact, 'boosted': boosted,
            'filled': filled, 'boosted_filled': boosted_filled}
    paths, statistics = [], {}
    for name, labels in maps.items():
        target = output_dir / f'{image_path.stem}_{name}.npy'
        np.save(target, labels, allow_pickle=False)
        paths.append(target)
        covered, pixels, coverage = pipeline.compute_coverage(labels)
        statistics[name] = {'label_count': int(len(np.unique(labels))),
                            'instance_count_excluding_zero': int(np.count_nonzero(np.unique(labels))),
                            'non_zero_px': int(covered), 'total_px': int(pixels),
                            'coverage_percent': float(coverage)}
    np.savez_compressed(output_dir / f'{image_path.stem}_raw_masks.npz', masks=masks, predicted_iou=scores)
    metadata = {'source_image': str(image_path.resolve()), 'n_sam_masks': len(masks),
                'n_coverage_masks': len(selected), 'selection_history': history,
                'statistics': statistics, 'max_distance': max_distance,
                'time_postprocess_sec': time.perf_counter() - start,
                'gif': str(gif) if gif else None,
                'selected_mask_indices': [item['original_mask_id'] for item in history]}
    (output_dir / f'{image_path.stem}_notebook_results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with (output_dir / f'{image_path.stem}_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', *statistics['sam']])
        writer.writeheader()
        for method, values in statistics.items():
            writer.writerow({'method': method, **values})
    render(image_path, paths, output_dir / f'{image_path.stem}_partitions.png')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--raw-masks', type=Path, help='NPZ with masks [N,H,W] and predicted_iou [N]')
    parser.add_argument('--uniform-iou', action='store_true', help='Use area-only selection if saved scores are missing')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--select-count', type=int, default=400)
    parser.add_argument('--max-distance', type=float, default=3)
    parser.add_argument('--create-gif', action='store_true')
    parser.add_argument('--save-step-images', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--sam2-config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'mps', 'auto'])
    parser.add_argument('--generate_at_least_n_masks', type=int, default=200)
    parser.add_argument('--mask-tuning-max-steps', type=int, default=3)
    args = parser.parse_args()
    tuning_history = []
    if args.raw_masks:
        with np.load(args.raw_masks, allow_pickle=False) as saved:
            masks = saved['masks']
            if 'predicted_iou' in saved:
                scores = saved['predicted_iou']
            elif args.uniform_iou:
                scores = np.ones(len(masks))
            else:
                parser.error('Saved masks lack predicted_iou; use --uniform-iou for area-only selection')
    else:
        import torch
        import generate_sam_partitions as sam
        from generate_coco_partitions import create_parser
        settings = create_parser().parse_args([])
        for name in ['checkpoint', 'sam2_config', 'device', 'generate_at_least_n_masks', 'mask_tuning_max_steps']:
            setattr(settings, name, getattr(args, name))
        generator, _, _ = sam.build_generator(settings)
        image = np.asarray(Image.open(args.image).convert('RGB'))
        with torch.inference_mode():
            annotations, tuning_history = sam.generate_at_least_n_masks(
                image, generator, min_masks=args.generate_at_least_n_masks,
                max_steps=args.mask_tuning_max_steps)
        masks = np.stack([item['segmentation'] for item in annotations]) if annotations else np.empty((0, *image.shape[:2]), dtype=bool)
        scores = np.array([item['predicted_iou'] for item in annotations])
    metadata = process(args.image, masks, scores, args.output_dir, args.select_count,
                       args.max_distance, args.create_gif, args.save_step_images)
    metadata.update({'mask_tuning_history': tuning_history, 'uniform_iou': args.uniform_iou,
                     'raw_masks_source': str(args.raw_masks) if args.raw_masks else None})
    (args.output_dir / f'{args.image.stem}_notebook_results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved notebook stages and previews to {args.output_dir}')


if __name__ == '__main__':
    main()
