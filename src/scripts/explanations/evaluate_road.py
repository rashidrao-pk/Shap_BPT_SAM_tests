#!/usr/bin/env python3
"""Add ROAD scores to a finished explanation run, reusing its saved attribution maps.

Use the same --config, --budget, and --bg-type as the explanation run. No explanation is
recomputed. Results go to <image-id>_road.json, and to road_results.csv and road_summary.csv
in the run's evaluation folder. Repeat the command to resume.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

import experiment_common as common
from model_adapters import load_adapter
import road

RUNNERS = list(common.DEFAULT_CONFIGS)
METRICS = ['morf', 'lerf', 'gap', 'morf_rel', 'lerf_rel', 'gap_rel']


def parser(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--runner', choices=RUNNERS, required=True)
    runner = preliminary.parse_known_args(argv)[0].runner
    result = common.create_parser(runner)
    result.description = __doc__
    result.add_argument('--runner', choices=RUNNERS, required=True)
    result.add_argument('--methods', nargs='+', help='Saved method names; default: every saved method')
    result.add_argument('--road-fractions', nargs='+', type=float, default=list(road.FRACTIONS))
    result.add_argument('--road-noise', type=float, default=road.NOISE)
    result.add_argument('--road-seed', type=int, default=0)
    result.add_argument('--road-short-side', type=int, default=road.SHORT_SIDE,
                        help='Downscale so the short side is at most this; 0 keeps full resolution')
    return result


def settings(args):
    return {'fractions': sorted(args.road_fractions), 'noise': args.road_noise, 'seed': args.road_seed,
            'short_side': args.road_short_side}


def image_seed(base, image_id):
    """Stable across runs and machines, unlike Python's hash()."""
    return (base + sum((index + 1) * ord(character) for index, character in enumerate(image_id))) % 2**32


def load_saved(folder, image_id, methods):
    attribution_path = folder / f'{image_id}_attributions.npz'
    prediction_path = folder / f'{image_id}_predictions.json'
    if not attribution_path.is_file() or not prediction_path.is_file():
        return None, None
    target = json.loads(prediction_path.read_text())['predicted_class_id']
    with np.load(attribution_path, allow_pickle=False) as saved:
        maps = {name: saved[name][0] for name in saved.files if not methods or name in methods}
    return target, maps


def evaluate_image(record, folder, adapter, config, methods=None):
    from PIL import Image
    image_id = record['image_id']
    target, maps = load_saved(folder, image_id, methods)
    if maps is None:
        return None
    path = folder / f'{image_id}_road.json'
    saved = json.loads(path.read_text()) if path.is_file() else {}
    done = saved.get('methods', {}) if saved.get('settings') == config else {}
    with Image.open(record['image_path']) as source:
        image = np.asarray(source.convert('RGB'))
    for method, heatmap in maps.items():
        if method in done:
            continue
        if heatmap.shape != image.shape[:2]:
            raise ValueError(f'{image_id} {method}: attribution {heatmap.shape} does not match image {image.shape[:2]}')
        started = time.perf_counter()
        done[method] = road.evaluate(adapter.predict_batch, image, heatmap, target, config['fractions'],
                                     config['noise'], image_seed(config['seed'], image_id), config['short_side'])
        done[method]['seconds'] = time.perf_counter() - started
        common.write_json(path, {'settings': config, 'image_id': image_id, 'target': target, 'methods': done})
        print(f"{image_id} {method}: MoRF {done[method]['morf']:.4f}, LeRF {done[method]['lerf']:.4f} "
              f"({done[method]['seconds']:.1f}s)", flush=True)
    return {method: done[method] for method in maps}


def write_tables(rows, folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'road_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['image_id', 'method', 'baseline', *METRICS], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in dict.fromkeys(row['method'] for row in rows):
        selected = [row for row in rows if row['method'] == method]
        item = {'method': method, 'images': len(selected)}
        for metric in METRICS:
            values = np.array([row[metric] for row in selected if row[metric] is not None], dtype=float)
            item[f'{metric}_mean'] = float(values.mean()) if values.size else None
            item[f'{metric}_std'] = float(values.std(ddof=1)) if values.size > 1 else None
        summary.append(item)
    with (folder / 'road_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def print_summary(summary):
    relative = all(item['gap_rel_mean'] is not None for item in summary)
    suffix = '_rel_mean' if relative else '_mean'
    print(f"\n{'Method':<24} {'Images':>6} {'MoRF (low)':>11} {'LeRF (high)':>12} {'Gap (high)':>11}")
    for item in sorted(summary, key=lambda row: -row['gap' + suffix]):
        print(f"{item['method']:<24} {item['images']:>6} {item['morf' + suffix]:>11.4f} "
              f"{item['lerf' + suffix]:>12.4f} {item['gap' + suffix]:>11.4f}")
    print('Scores relative to the unmodified image.' if relative else 'Raw model scores.')


def run(args, adapter=None):
    kind = args.runner
    inputs = common.prepare_inputs(kind, args)
    run_dir = inputs['run_dir']
    if not run_dir.is_dir():
        raise FileNotFoundError(f'{run_dir}: run the explanations first, with the same --budget and --bg-type')
    print(f"ROAD: {len(inputs['records'])} images from {run_dir}", flush=True)
    if args.dry_run:
        return []
    if adapter is None:
        import torch
        name = args.device or inputs['config'].get('device', 'auto')
        if name == 'auto':
            name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        adapter = load_adapter(kind, args, torch.device(name))
    config = settings(args)
    rows, skipped = [], []
    for record in inputs['records']:
        results = evaluate_image(record, run_dir / record['image_id'], adapter, config, args.methods)
        if results is None:
            skipped.append(record['image_id'])
            continue
        rows.extend({'image_id': record['image_id'], 'method': method, **result} for method, result in results.items())
    if skipped:
        print(f"Skipped {len(skipped)} images without saved attributions: {', '.join(skipped[:5])}", flush=True)
    if not rows:
        raise ValueError(f'No saved attributions found under {run_dir}')
    summary = write_tables(rows, inputs['evaluation_dir'])
    print_summary(summary)
    print(f"Saved: {Path(inputs['evaluation_dir']) / 'road_summary.csv'}", flush=True)
    return summary


if __name__ == '__main__':
    run(parser().parse_args())
