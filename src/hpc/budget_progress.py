#!/usr/bin/env python3
"""Read-only progress for budget studies, including partially computed images."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import time


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def duration(seconds):
    if seconds is None:
        return 'warming up'
    minutes = int(seconds // 60)
    return f'{minutes//60}h {minutes%60:02d}m'


def snapshot(task, home):
    folder = Path(task['output'])
    study = read_json(folder/'study.json')
    if not study:
        return {'task': task['id'], 'text': f"{task['id']}: waiting for study initialization"}
    ids = study['splits']['calibration'] + study['splits']['validation']
    methods = study['methods']
    budgets = study['budgets'] + [study['reference_budget']]
    repeats = study['repeats']
    per_image = len(methods)*len(budgets)*repeats
    total = len(ids)*per_image
    saved, completed, current = 0, 0, None
    timings, counts, modified = defaultdict(list), defaultdict(int), []
    for image in ids:
        image_saved = 0
        for method in methods:
            directory = folder/image/method.replace('+', '_')
            for budget in budgets:
                for repeat in range(repeats):
                    name = f'reference_s{repeat}' if budget == study['reference_budget'] else f'B{budget}_s{repeat}'
                    marker = directory/(name+'.json')
                    row = read_json(marker)
                    if (not row or row.get('signature') != study['signature'] or
                            not marker.with_suffix('.npz').is_file()):
                        continue
                    saved += 1; image_saved += 1
                    key = (method, budget)
                    counts[key] += 1
                    timings[key].append(float(row['explanation_sec'])+float(row['evaluation_sec']))
                    modified.append(marker.stat().st_mtime)
        completed += image_saved == per_image
        if current is None and image_saved < per_image:
            current = image
    expected_groups = [(method, budget) for method in methods for budget in budgets]
    remaining = (sum(max(0, len(ids)*repeats-counts[key])*statistics.median(timings[key])
                     for key in expected_groups) if all(timings[key] for key in expected_groups) else None)
    log = home/'logs'/(task['id']+'.log')
    try:
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-16384))
            tail = stream.read().decode(errors='replace')
        phases = re.findall(r'(\d+x\d+) (.+?): B=(\d+), seed=(\d+)', tail)
        phase = (f'{phases[-1][1]}, B={phases[-1][2]}, seed={phases[-1][3]}, {phases[-1][0]}'
                 if phases else 'preparing inputs/tree or finalizing results')
    except OSError:
        phase = 'not logged yet'
    age = max(0, time.time()-max(modified)) if modified else None
    status = read_json(home/'task_status'/(task['id']+'.json'), {})
    elapsed = None
    if status.get('started_utc'):
        elapsed = time.time()-datetime.fromisoformat(status['started_utc']).timestamp()
    if study['state'] == 'complete':
        phase = 'all image evaluations complete; see pilot report'
    label = task['id'].removeprefix('budget_')
    text = (f"{label} | GPU {status.get('gpu', '?')} | images cached {completed}/{len(ids)} "
            f"| evaluations saved {saved}/{total} ({100*saved/total:.1f}%)\n"
            f"  Current image: {current or 'all cached'} | Last logged phase: {phase}\n"
            f"  Session elapsed: {duration(elapsed)} | Last save: {duration(age)} ago "
            f"| Approx. remaining evaluation time: {duration(remaining)}")
    return {'task': task['id'], 'saved': saved, 'total': total, 'images_cached': completed,
            'images': len(ids), 'current_image': current, 'phase': phase,
            'eta_evaluation_seconds': remaining, 'text': text}


def report(manifest_path):
    home = manifest_path.resolve().parent
    manifest = read_json(manifest_path)
    if not manifest:
        raise ValueError(f'Cannot read manifest: {manifest_path}')
    rows = [snapshot(task, home) for task in manifest['tasks'] if task['type'] == 'budget']
    header = datetime.now(timezone.utc).strftime('[%Y-%m-%d %H:%M:%S UTC] BUDGET PROGRESS')
    return header+'\n'+'\n'.join(row['text'] for row in rows)+(
        '\nETA uses saved median times by method/budget; excludes tree building, mean-reference AUC and reports.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_root', type=Path)
    parser.add_argument('--manifest', help='Default: active_manifest.txt, then manifest.json')
    parser.add_argument('--watch', type=float, default=0, help='Refresh interval in seconds; default: print once')
    parser.add_argument('--append-log', type=Path, help='Also append progress to the main Slurm output')
    args = parser.parse_args()
    active = args.study_root/'active_manifest.txt'
    selected = args.manifest or (active.read_text().strip() if active.is_file() else 'manifest.json')
    while True:
        message = report(args.study_root/selected)
        print(message, flush=True)
        if args.append_log:
            with args.append_log.open('a') as stream:
                stream.write(message+'\n')
        if args.watch <= 0:
            break
        time.sleep(max(1., args.watch))
