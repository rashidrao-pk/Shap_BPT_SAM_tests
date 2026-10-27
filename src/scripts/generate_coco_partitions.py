#!/usr/bin/env python3
"""Generate resumable MS-COCO partition files consumed by run_yolo_full.py."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import yaml

import generate_sam_partitions as sam


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='MSCOCO_epito')
    parser.add_argument('--model', choices=['sam1', 'sam2'], default='sam2')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--sam2-config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--sam1-variant', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'], default='cuda')
    parser.add_argument('--max-regions', type=int, default=62)
    parser.add_argument('--min-region-pixels', type=int, default=64)
    parser.add_argument('--points-per-side', type=int, default=16)
    parser.add_argument('--points-per-batch', type=int, default=16)
    parser.add_argument('--pred-iou-thresh', type=float, default=0.7)
    parser.add_argument('--stability-score-thresh', type=float, default=0.85)
    parser.add_argument('--limit', type=int, help='Generate the first N images; omit for all images')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate completed images')
    parser.add_argument('--dry-run', action='store_true', help='Check paths and list work without loading SAM')
    return parser.parse_args()


def run(args):
    if args.max_regions < 1 or args.min_region_pixels < 1:
        raise ValueError('Region limits must be positive')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.suffix:
        config_path = root / 'src/configs' / f'{args.config}.yaml'
    elif not config_path.is_absolute():
        config_path = root / config_path
    config = yaml.safe_load(config_path.read_text())
    data = config['data']
    output = Path(data['masks_path']) / data['mask_dir'] / data['mask_dir_final']
    with Path(data['annotation_file']).open() as stream:
        images = sorted(json.load(stream)['images'], key=lambda item: int(item['id']))
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise ValueError('No images selected')
    print(f'Images: {len(images)}; model: {args.model}; output: {output}', flush=True)
    pending = []
    for image in images:
        path = Path(data['image_dir']) / image['file_name']
        if not path.is_file():
            raise FileNotFoundError(path)
        image_id = f"{int(image['id']):012d}"
        metadata_path = output / f'{image_id}_results.json'
        completed = all((output / f'{image_id}_{name}.npy').is_file()
                        for name in ('sam', 'refined')) and metadata_path.is_file()
        if completed and not args.overwrite:
            previous = json.loads(metadata_path.read_text())
            settings = {key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                        if key not in {'limit', 'overwrite', 'dry_run', 'config'}}
            if previous.get('generation_settings') != settings:
                raise ValueError(f'{image_id}: settings differ; use a new output folder or --overwrite')
            continue
        pending.append((image_id, path))
    print(f'Pending: {len(pending)}; already complete: {len(images) - len(pending)}', flush=True)
    if args.dry_run:
        for image_id, path in pending[:10]:
            print(f'{image_id}: {path}')
        return
    if not pending:
        return
    loaded = sam.build_generator(args)
    output.mkdir(parents=True, exist_ok=True)
    for index, (image_id, path) in enumerate(pending, 1):
        start = time.perf_counter()
        image = np.asarray(Image.open(path).convert('RGB'))
        masks, seconds, device, checkpoint = sam.infer_masks(image, args, loaded)
        refinement_start = time.perf_counter()
        raw = sam.enforce_region_budget(sam.raw_overwrite(masks), args.max_regions)
        refined = sam.refined_partitions(masks, args.min_region_pixels, args.max_regions)
        refinement_seconds = time.perf_counter() - refinement_start
        for name, labels in [('sam', raw), ('refined', refined)]:
            target = output / f'{image_id}_{name}.npy'
            temporary = target.with_suffix('.npy.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, labels)
            temporary.replace(target)
        metadata = {
            'name': path.name, 'model': args.model, 'checkpoint': checkpoint,
            'device': device, 'n_sam_masks': len(masks),
            'n_refined_instances': len(np.unique(refined)),
            'time_sam': seconds, 'time_refined': refinement_seconds,
            'total_time_sec': time.perf_counter() - start,
            'generation_settings': {key: str(value) if isinstance(value, Path) else value
                                    for key, value in vars(args).items()
                                    if key not in {'limit', 'overwrite', 'dry_run', 'config'}},
        }
        target = output / f'{image_id}_results.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(target)
        print(f'[{index}/{len(pending)}] {image_id}: {len(masks)} masks, '
              f'{metadata["n_refined_instances"]} regions, '
              f'{metadata["total_time_sec"]:.1f}s', flush=True)


if __name__ == '__main__':
    run(parse_args())
