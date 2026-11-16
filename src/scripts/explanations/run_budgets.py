#!/usr/bin/env python3
"""Run one XAI command at several budgets, one budget after another, resumably.

    python src/scripts/explanations/run_budgets.py --budgets 3000 5000 -- \\
        src/scripts/explanations/OD_yolo.py --config MSCOCO_epito_sam1 --show-gt --explain_all --no-heatmaps

Each budget is the same runner command plus --budget B, so each writes its own _B<B> folder.
A budget is marked done only when the runner exits cleanly. Repeat the same command after a
disconnect or time limit: done budgets are skipped and the runner skips its completed images.
State (attempts, times, done markers) is kept in results/budget_runs/<runner>_<hash>/.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN = {'--budget', '--max-evals', '--no-resume', '--bpt-budgets', '--partition-budgets', '--budget-selection'}
MAX_FAILURES = 2  # Consecutive crashes of one budget (not interruptions) before the chain stops.


def parse(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' not in argv:
        raise SystemExit('Separate the runner command with --, e.g. run_budgets.py --budgets 3000 5000 -- OD_yolo.py ...')
    split = argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--budgets', nargs='+', type=int, required=True, help='Run in this order')
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'results' / 'budget_runs')
    parser.add_argument('--dry-run', action='store_true', help='Show the plan and what is done; run nothing')
    args = parser.parse_args(argv[:split])
    args.command = argv[split + 1:]
    if not args.command or not args.command[0].endswith('.py'):
        raise SystemExit('After -- give the runner script, e.g. src/scripts/explanations/OD_yolo.py')
    clash = sorted({item.split('=')[0] for item in args.command} & FORBIDDEN)
    if clash:
        raise SystemExit(f'Remove {", ".join(clash)} from the runner command; budgets come from --budgets and resume must stay on')
    if any(budget < 2 for budget in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        raise SystemExit('--budgets must be distinct and at least 2')
    return args


def state_folder(args):
    """One folder per runner command, so different experiments never share done markers."""
    key = hashlib.sha256(json.dumps(args.command).encode()).hexdigest()[:10]
    return args.state_dir / f'{Path(args.command[0]).stem}_{key}'


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)  # Never leaves a half-written file after a kill.


def duration(seconds):
    hours, rest = divmod(round(seconds), 3600)
    return f'{hours:02d}:{rest // 60:02d}:{rest % 60:02d}'


def summary(folder, budgets):
    lines = [f'{"Budget":>7}  {"Status":<11} {"Attempts":>8} {"Time":>9}']
    for budget in budgets:
        attempts = read_json(folder / f'B{budget}_attempts.json', [])
        status = 'done' if (folder / f'B{budget}.done').is_file() else 'started' if attempts else 'pending'
        seconds = sum(item.get('seconds', 0) for item in attempts)
        lines.append(f'{budget:>7}  {status:<11} {len(attempts):>8} {duration(seconds):>9}')
    return '\n'.join(lines)


def cancel_chain():
    job = os.environ.get('CHAIN_NEXT_JOB')
    if job:
        subprocess.run(['scancel', job], check=False)
        print(f'Cancelled follow-up job {job}', flush=True)


class Interrupted(Exception):
    pass


def run_budget(command, budget, folder):
    attempts_path = folder / f'B{budget}_attempts.json'
    attempts = read_json(attempts_path, [])
    attempt = {'started': datetime.now().isoformat(timespec='seconds'), 'job': os.environ.get('SLURM_JOB_ID'),
               'host': os.uname().nodename}
    attempts.append(attempt)
    write_json(attempts_path, attempts)
    print(f'\n=== Budget {budget}: attempt {len(attempts)} ===\n{" ".join(command)}', flush=True)
    started = time.perf_counter()
    child = subprocess.Popen(command, cwd=ROOT)

    def stop(signum, _frame):  # Slurm time limit or Ctrl-C: let the runner stop, then record it.
        child.send_signal(signum)
        raise Interrupted(signal.Signals(signum).name)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        code = child.wait()
        outcome = 'done' if code == 0 else 'failed'
    except Interrupted as reason:
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            child.kill()
        code, outcome = None, f'interrupted ({reason})'
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    attempt.update(ended=datetime.now().isoformat(timespec='seconds'), seconds=time.perf_counter() - started,
                   exit_code=code, outcome=outcome)
    write_json(attempts_path, attempts)
    if outcome == 'done':
        write_json(folder / f'B{budget}.done', {'budget': budget, 'command': command, 'finished': attempt['ended'],
                                                 'attempts': len(attempts),
                                                 'seconds': sum(item.get('seconds', 0) for item in attempts)})
    return outcome, attempts


def main(argv=None):
    args = parse(argv)
    folder = state_folder(args)
    write_json(folder / 'plan.json', {'command': args.command, 'budgets': args.budgets})
    print(f'State: {folder}\n{summary(folder, args.budgets)}', flush=True)
    if args.dry_run:
        return 0
    for budget in args.budgets:
        if (folder / f'B{budget}.done').is_file():
            print(f'Budget {budget}: done earlier, skipped', flush=True)
            continue
        command = [sys.executable, *args.command, '--budget', str(budget)]
        outcome, attempts = run_budget(command, budget, folder)
        if outcome.startswith('interrupted'):
            print(f'Budget {budget} {outcome}; repeat the same command to resume.', flush=True)
            return 143
        if outcome == 'failed':
            crashes = 0
            for item in reversed(attempts):
                if item.get('outcome') != 'failed':
                    break
                crashes += 1
            print(f'Budget {budget} failed (exit {attempts[-1]["exit_code"]}); see the runner output above.', flush=True)
            if crashes >= MAX_FAILURES:
                print(f'{crashes} failures in a row: stopping instead of retrying. Fix the error, then repeat the command.', flush=True)
                cancel_chain()
            return 1
    print(f'\nAll budgets done.\n{summary(folder, args.budgets)}', flush=True)
    cancel_chain()
    return 0


if __name__ == '__main__':
    sys.exit(main())
