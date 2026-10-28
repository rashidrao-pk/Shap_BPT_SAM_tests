#!/usr/bin/env python3
"""Generate resumable MS-COCO partition files consumed by run_yolo_full.py."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import sys
import traceback
import json
import re
from pathlib import Path
import time

import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm

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
    parser.add_argument('--generate_at_least_n_masks', '--generate-at-least-n-masks',
                        type=int, nargs='?', const=400, default=None,
                        help='Retry threshold tuning toward N raw masks (bare flag: 400)')
    parser.add_argument('--mask-tuning-max-steps', type=int, default=3,
                        help='Maximum mask-generation attempts per image (default: 3)')
    # SAM parameters
    parser.add_argument('--min-region-pixels', type=int, default=64)
    # parser.add_argument('--points-per-side', type=int, default=16)
    # parser.add_argument('--points-per-batch', type=int, default=16)
    # parser.add_argument('--pred-iou-thresh', type=float, default=0.7)
    # parser.add_argument('--stability-score-thresh', type=float, default=0.85)
        # SAM automatic mask generator parameters; see SAM docs for details.
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.6)
    # 
    parser.add_argument("--stability-score-thresh", type=float, default=0.6)
    parser.add_argument("--stability-score-offset", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-overlap-ratio", type=float, default=512/1500)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--point-grids", type=float, default=None)
    parser.add_argument("--min-mask-region-area", type=float, default=0)
    parser.add_argument("--output-mode", type=str, default="binary_mask")
    parser.add_argument("--use-m2m", type=bool, default=True)
    # 
    parser.add_argument('--limit', type=int, help='Generate the first N images; omit for all images')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate completed images')
    parser.add_argument('--log-file', type=Path, help='Append log; default: <masks_path>/<mask_dir>/<partition-folder-name>.log')
    parser.add_argument('--dry-run', action='store_true', help='Check paths and list work without loading SAM')
    return parser.parse_args()


def partition_folder_name(data, args):
    """Describe the dataset/model/region budget/raw-mask target in the path."""
    dataset = re.sub(r'[^A-Za-z0-9-]', '', data.get('name', 'MSCOCO'))
    if not dataset:
        raise ValueError('Dataset name must contain letters or digits')
    if args.model == 'sam1':
        variant = args.sam1_variant
    else:
        checkpoint = getattr(args, 'checkpoint', None)
        stem = Path(checkpoint).stem if checkpoint else ''
        variants = {'tiny': 'tiny', 'small': 'small', 'base_plus': 'base_plus', 'large': 'large'}
        variant = next((value for suffix, value in variants.items()
                        if stem.endswith('hiera_' + suffix)), None)
        if variant is None:
            config = Path(getattr(args, 'sam2_config', 'sam2.1_hiera_t.yaml')).stem
            variant = {'t': 'tiny', 's': 'small', 'b+': 'base_plus', 'l': 'large'}.get(config.rsplit('_', 1)[-1])
        if variant is None:
            raise ValueError('Cannot determine SAM2 model size from checkpoint/config')
    target = getattr(args, 'generate_at_least_n_masks', None)
    return f'{dataset}_{args.model}_{variant}_{args.max_regions}_refined_{target if target is not None else "singlepass"}'


def _run(args, progress_stream):
    mask_target = getattr(args, 'generate_at_least_n_masks', None)
    if mask_target is not None and (mask_target < 1 or args.mask_tuning_max_steps < 1):
        raise ValueError('Mask target and tuning attempt count must be positive')
    if getattr(args, 'output_mode', 'binary_mask') != 'binary_mask':
        raise ValueError('Partition generation requires --output-mode binary_mask')
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
    folder_name = partition_folder_name(data, args)
    output = Path(data['masks_path']) / data['mask_dir'] / folder_name
    with Path(data['annotation_file']).open() as stream:
        images = sorted(json.load(stream)['images'], key=lambda item: int(item['id']))
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise ValueError('No images selected')
    print(f'Images: {len(images)}; model: {args.model}; output: {output}', flush=True)
    print(f'For XAI set data.mask_dir_final: "{folder_name}"', flush=True)
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
                        if key not in {'limit', 'overwrite', 'dry_run', 'config', 'log_file'}}
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
    progress = tqdm(pending, total=len(images), initial=len(images) - len(pending),
                    desc=f"{args.model.upper()} MSCOCO", ascii=True,
                    dynamic_ncols=True, file=progress_stream)
    for index, (image_id, path) in enumerate(progress, 1):
        start = time.perf_counter()
        image = np.asarray(Image.open(path).convert('RGB'))
        tuning_history = []
        if mask_target is not None:
            masks, seconds, device, checkpoint, tuning_history = sam.infer_masks(
                image, args, loaded, return_history=True)
        else:
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
            'mask_tuning_history': tuning_history,
            'mask_target_met': len(masks) >= mask_target if mask_target is not None else None,
            'time_sam': seconds, 'time_refined': refinement_seconds,
            'total_time_sec': time.perf_counter() - start,
            'generation_settings': {key: str(value) if isinstance(value, Path) else value
                                    for key, value in vars(args).items()
                                    if key not in {'limit', 'overwrite', 'dry_run', 'config', 'log_file'}},
        }
        target = output / f'{image_id}_results.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(target)
        print(f'{datetime.now(timezone.utc).isoformat()} [{index}/{len(pending)}] {image_id}: {len(masks)} masks, '
              f'{metadata["n_refined_instances"]} regions, '
              f'sam_s={seconds:.3f} refinement_s={refinement_seconds:.3f} '
              f'total_s={metadata["total_time_sec"]:.3f}', flush=True)
        progress.set_postfix(masks=len(masks), regions=metadata['n_refined_instances'],
                             seconds=f"{metadata['total_time_sec']:.1f}")


def run(args):
    if args.dry_run:
        return _run(args, sys.stderr)
    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.suffix:
        config_path = root / 'src/configs' / f'{args.config}.yaml'
    elif not config_path.is_absolute():
        config_path = root / config_path
    data = yaml.safe_load(config_path.read_text())['data']
    log_path = getattr(args, 'log_file', None) or (
        Path(data['masks_path']) / data['mask_dir'] / f'{partition_folder_name(data, args)}.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    terminal = sys.stderr
    with log_path.open('a', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            start = time.perf_counter()
            print(f"\n{datetime.now(timezone.utc).isoformat()} START config={args.config} "
                  f"model={args.model} checkpoint={args.checkpoint}", flush=True)
            try:
                _run(args, terminal)
            except BaseException:
                traceback.print_exc(file=log)
                print(f"{datetime.now(timezone.utc).isoformat()} FAILED "
                      f"elapsed_s={time.perf_counter() - start:.3f}", flush=True)
                raise
            print(f"{datetime.now(timezone.utc).isoformat()} COMPLETE "
                  f"elapsed_s={time.perf_counter() - start:.3f}", flush=True)


if __name__ == '__main__':
    run(parse_args())
