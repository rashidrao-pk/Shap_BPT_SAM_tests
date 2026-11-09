#!/usr/bin/env python3
"""Add ROAD scores to a finished explanation run, reusing its saved attribution maps.

Use the same --config, --budget, and --bg-type as the explanation run. No explanation is
recomputed. Results go to <image-id>_road.json, and to road_results.csv and road_summary.csv
in the run's evaluation folder. The scores are also added as road_* columns to
auc_results_all_images_current_run.csv, in both the run and evaluation folders, so one table
holds every metric. Repeat the command to resume.
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
    return result


settings = road.settings
image_seed = road.image_seed


def load_saved(folder, image_id, methods):
    attribution_path = folder / f'{image_id}_attributions.npz'
    prediction_path = folder / f'{image_id}_predictions.json'
    if not attribution_path.is_file() or not prediction_path.is_file():
        return None, None
    target = json.loads(prediction_path.read_text())['predicted_class_id']
    with np.load(attribution_path, allow_pickle=False) as saved:
        maps = {name: saved[name][0] for name in saved.files if not methods or name in methods}
    return target, maps


def score_maps(predict, image, maps, target, image_id, config, path, verbose=True):
    """ROAD for each map, reusing the scores already in path when they used the same settings.

    Saves after every method, so an interrupted run resumes where it stopped.
    Returns {method: scores} for the given maps.
    """
    saved = json.loads(path.read_text()) if path.is_file() else {}
    done = saved.get('methods', {}) if saved.get('settings') == config else {}
    for method, heatmap in maps.items():
        if method in done:
            continue
        heatmap = np.asarray(heatmap)
        if heatmap.shape != image.shape[:2]:
            raise ValueError(f'{image_id} {method}: attribution {heatmap.shape} does not match image {image.shape[:2]}')
        started = time.perf_counter()
        done[method] = road.evaluate(predict, image, heatmap, target, config['fractions'],
                                     config['noise'], image_seed(config['seed'], image_id), config['short_side'])
        done[method]['seconds'] = time.perf_counter() - started
        common.write_json(path, {'settings': config, 'image_id': image_id, 'target': target, 'methods': done})
        if verbose:
            print(f"{image_id} {method}: MoRF {done[method]['morf']:.4f}, LeRF {done[method]['lerf']:.4f} "
                  f"({done[method]['seconds']:.1f}s)", flush=True)
    return {method: done[method] for method in maps}


def road_columns(scores):
    """The road_<metric> columns of one method's ROAD scores, as added to the AUC table."""
    return {f'road_{metric}': scores.get(metric) for metric in METRICS}


def saved_columns(path, config):
    """{method: road columns} from a saved <image-id>_road.json, if it used these settings."""
    saved = json.loads(path.read_text()) if path.is_file() else {}
    if saved.get('settings') != config:
        return {}
    return {method: road_columns(scores) for method, scores in saved.get('methods', {}).items()}


def evaluate_image(record, folder, adapter, config, methods=None):
    from PIL import Image
    image_id = record['image_id']
    target, maps = load_saved(folder, image_id, methods)
    if maps is None:
        return None
    with Image.open(record['image_path']) as source:
        image = np.asarray(source.convert('RGB'))
    if hasattr(adapter, 'set_record'):
        adapter.set_record(record)  # VL_RefCOCO: score against this record's expression.
    return score_maps(adapter.predict_batch, image, maps, target, image_id, config,
                      folder / f'{image_id}_road.json')


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


def merge_into_auc_table(rows, path):
    """Add road_<metric> columns to an AUC table, matching rows by image_id and method.

    Rows that ROAD did not score this time keep any road values they already had, so a run
    with --limit or --methods adds to the table instead of clearing it. Every other column is
    kept exactly as written. Returns the number of AUC rows that received ROAD scores, or None
    when the table does not exist.
    """
    import pandas as pd
    if not path.is_file():
        return None
    # Read as text so ids such as 000000076211 and every number stay exactly as written.
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if table.empty or not {'image_id', 'method'} <= set(table.columns):
        return 0
    scores = {(str(row['image_id']), row['method']): row for row in rows}
    keys = list(zip(table['image_id'], table['method']))
    matched = [key in scores for key in keys]
    for metric in METRICS:
        column = f'road_{metric}'
        if column not in table:
            table[column] = ''
        values = table[column].astype(object)
        for index, key in enumerate(keys):
            if matched[index]:
                value = scores[key].get(metric)
                values.iat[index] = '' if value is None else repr(float(value))
        table[column] = values
    temporary = path.with_name(path.name + '.tmp')
    table.to_csv(temporary, index=False)
    temporary.replace(path)
    return sum(matched)


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
    for folder in dict.fromkeys([Path(inputs['evaluation_dir']), Path(run_dir)]):
        table = folder / 'auc_results_all_images_current_run.csv'
        merged = merge_into_auc_table(rows, table)
        if merged is None:
            print(f'No AUC table to extend at {table}', flush=True)
        else:
            print(f'Added ROAD to {merged} of {len(rows)} scored rows in {table}', flush=True)
    print_summary(summary)
    print(f"Saved: {Path(inputs['evaluation_dir']) / 'road_summary.csv'}", flush=True)
    return summary


if __name__ == '__main__':
    run(parser().parse_args())
