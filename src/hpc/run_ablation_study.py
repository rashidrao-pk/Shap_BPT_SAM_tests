#!/usr/bin/env python3
"""Run a frozen ablation task manifest on allocated GPUs, preserving completed tasks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from budget_progress import snapshot


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def task_complete(task):
    directory = Path(task['output'])
    if task['type'] == 'auc':
        status = directory / 'comparison_status.json'
        if not status.exists() or json.loads(status.read_text()).get('state') != 'complete':
            return False
        names = ['summary.json', 'conclusion.md', 'measurements.csv', 'paired_comparisons.csv',
                 'auc_resolution_comparison.png']
    else:
        status = directory / 'study.json'
        if not status.exists() or json.loads(status.read_text()).get('state') != 'complete':
            return False
        names = ['selection.json', 'conclusion.md', 'comparisons.json', 'budget_comparisons.csv',
                 'reference_diagnostics.json', 'budget_report.html']
    return all((directory / name).is_file() for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    home = args.manifest.resolve().parent
    lock = (home / 'controller.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another controller is already running for this study')
    code = home / 'code'
    if manifest.get('budget_protocol', {}).get('background') == 'full' and not (home / 'preflight_complete.json').is_file():
        # Validate every scope on the compute node before spending GPU time on the first pilot.
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_XET='1')
        for task in manifest['tasks']:
            log_path = home / 'logs' / (task['id'] + '_preflight.log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"PREFLIGHT {task['id']}", flush=True)
            with log_path.open('w') as log:
                subprocess.run([sys.executable, str(code / task['script']), *task['args'], '--dry-run'],
                               cwd=code, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(home / 'preflight_complete.json', {'state': 'complete', 'tasks': len(manifest['tasks'])})
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    work = queue.Queue()
    for task in manifest['tasks']:
        if task_complete(task):
            print(f"SKIP {task['id']}: verified complete", flush=True)
        else:
            work.put(task)
    failures = []
    def worker(device):
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED='1',
            HF_HUB_DISABLE_XET='1', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='6',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='6', MPLCONFIGDIR=str(home / 'mpl' / device))
        environment.pop('PYTHONPATH', None)
        while True:
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            started = time.time()
            status_path = home / 'task_status' / (task['id'] + '.json')
            command = [sys.executable, '-u', str(code / task['script']), *task['args']]
            write_json(status_path, {'state': 'running', 'task': task['id'], 'gpu': device,
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'started_utc': datetime.now(timezone.utc).isoformat(),
                'command': command, 'output': task['output']})
            print(f"START {task['id']} GPU={device}", flush=True)
            log_path = home / 'logs' / (task['id'] + '.log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('a', buffering=1) as log:
                log.write('\nSTART ' + datetime.now(timezone.utc).isoformat() + '\n')
                process = subprocess.Popen(command, cwd=code, env=environment, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    try:
                        returncode = process.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        if task['type'] == 'budget':
                            try:
                                print(snapshot(task, home)['text'], flush=True)
                            except Exception as error:
                                print(f"PROGRESS {task['id']}: unavailable ({error}); inference continues", flush=True)
                result = subprocess.CompletedProcess(command, returncode)
            success = result.returncode == 0 and task_complete(task)
            write_json(status_path, {'state': 'complete' if success else 'failed', 'task': task['id'],
                'gpu': device, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'returncode': result.returncode,
                'elapsed_seconds': time.time()-started, 'output': task['output'], 'log': str(log_path)})
            if not success:
                failures.append(task['id'])
            print(f"{'DONE' if success else 'FAILED'} {task['id']}: {time.time()-started:.1f}s", flush=True)
            work.task_done()
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker, devices))
    write_json(home / 'execution_status.json', {'state': 'failed' if failures else 'complete',
        'failures': failures, 'verified_complete': [task['id'] for task in manifest['tasks'] if task_complete(task)],
        'total_tasks': len(manifest['tasks'])})
    if failures:
        raise SystemExit(1)
    if manifest.get('postprocess'):
        subprocess.run([sys.executable, str(code / manifest['postprocess']), str(home),
                        *manifest.get('postprocess_args', [])],
                       cwd=code, check=True)


if __name__ == '__main__':
    main()
