#!/usr/bin/env python3
"""Summarise the per-image <id>_timing.json files of an XAI run: where the time goes and what is GPU, CPU or disk.

    python src/scripts/explanations/timing_report.py \
      results/explanations/OD_yolo/MSCOCO_sam2_large_254_compact-sam_200_split/bg_full_B1000

Prints the GPU/CPU/IO split, the steps with the largest own time per image (what to speed up first), and writes
timing_report.csv / timing_report.md next to the images. Steps nested under a method (e.g.
method:BPT-1000/yolo.predict/inference (GPU)) are folded into a generic name (*/yolo.predict/inference (GPU)) so
that the same work is added up across methods.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HINTS = {
    'inference (GPU)': 'GPU forward: larger --eval-batch-size / --auc-batch-size, half precision, or a faster GPU',
    'model.predict_batch (incl. preprocessing)': 'model call: larger batches; move preprocessing to the GPU',
    'preprocess (CPU)': 'Ultralytics letterbox and to-tensor on the CPU: pass tensors already on the GPU',
    'postprocess NMS (CPU)': 'NMS on the CPU: lower max_det or raise conf',
    'compose_masked_images': 'np.where per mask x background on the CPU: compose on the GPU with torch',
    'plots.single_maps': 'matplotlib: --no-heatmaps, or plot only some images',
    'plots.grids_and_curves': 'matplotlib: --no-plots for bulk runs',
    'plots.grids': 'matplotlib: --no-plots for bulk runs',
    'plots.auc_curves': 'matplotlib: --no-plots for bulk runs',
    'build_bpt': 'BPT construction (Cython, CPU): built once per image already',
    'build_bpt_guided': 'guided BPT construction (Cython, CPU)',
    'auc_metrics': 'AUC curves: fewer --auc-num-samples',
    'road_metrics': 'ROAD imputation on the CPU',
}


def generic(step):
    parts = step.split('/')
    return '/'.join(['*' if part.startswith('method:') else part for part in parts])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', type=Path, help='Folder with one subfolder per image')
    parser.add_argument('--top', type=int, default=15)
    args = parser.parse_args(argv)
    files = sorted(args.run_dir.glob('*/*_timing.json'))
    if not files:
        raise SystemExit(f'No <id>_timing.json under {args.run_dir}; rerun the images with the current code')
    runs = [json.loads(path.read_text()) for path in files]
    n = len(runs)
    total = max(sum(run['total_seconds'] for run in runs), 1e-9)
    kinds = defaultdict(float)
    for run in runs:
        for kind, value in run['by_kind_seconds'].items():
            kinds[kind] += value
    own, calls, kind_of = defaultdict(float), defaultdict(int), {}
    methods = defaultdict(float)
    for run in runs:
        for step in run['steps']:
            key = generic(step['step'])
            own[key] += step['self_seconds']
            calls[key] += step['calls']
            kind_of[key] = step['kind']
            if '/' not in step['step'] and step['step'].startswith('method:'):
                methods[step['step']] += step['seconds']
    first = runs[0]
    lines = [f"# Timing: {args.run_dir}", '',
             f"{n} images, {total / n:.1f} s per image on {first.get('gpu') or first.get('device')} "
             f"(host {first.get('host')}), budget {first.get('budget')}, batch {first.get('eval_batch_size')}.", '',
             '| Kind | Seconds per image | Share |', '| --- | --- | --- |']
    lines += [f'| {kind} | {value / n:.2f} | {value / total:.0%} |' for kind, value in sorted(kinds.items(), key=lambda x: -x[1])]
    images = sum(run['counters'].get('model_images', 0) for run in runs)
    if images and kinds.get('gpu'):
        lines += ['', f"Model images: {images / n:.0f} per image, {images / kinds['gpu']:.0f} per GPU second."]
    lines += ['', f'## Steps with the most own time (top {args.top})', '',
              '| Step | Kind | Own s / image | Share | Calls / image | What to try |', '| --- | --- | --- | --- | --- | --- |']
    rows = []
    for key, value in sorted(own.items(), key=lambda x: -x[1]):
        hint = HINTS.get(key.split('/')[-1], '')
        rows.append({'step': key, 'kind': kind_of[key], 'own_seconds_per_image': value / n, 'share': value / total,
                     'calls_per_image': calls[key] / n, 'hint': hint})
    for row in rows[:args.top]:
        lines.append(f"| {row['step']} | {row['kind']} | {row['own_seconds_per_image']:.2f} | {row['share']:.0%} | "
                     f"{row['calls_per_image']:.0f} | {row['hint']} |")
    lines += ['', '## Methods (total time per image, model calls included)', '',
              '| Method | Seconds / image |', '| --- | --- |']
    lines += [f'| {name[7:]} | {value / n:.2f} |' for name, value in sorted(methods.items(), key=lambda x: -x[1])]
    text = '\n'.join(lines) + '\n'
    (args.run_dir / 'timing_report.md').write_text(text)
    import csv
    with (args.run_dir / 'timing_report.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(text)
    print(f"Saved: {args.run_dir / 'timing_report.md'}, timing_report.csv")


if __name__ == '__main__':
    main()
