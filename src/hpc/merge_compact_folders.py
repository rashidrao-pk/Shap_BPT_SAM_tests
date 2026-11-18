#!/usr/bin/env python3
"""Merge partition folders that carry the compact strategy in their name into one folder per SAM setting.

    <base>_greedy/<id>_compact.npy    ->  <base>/<id>_compact_greedy.npy
    <base>_mso_0.5/<id>_compact.npy   ->  <base>/<id>_compact_mso_0.5.npy
    <base>_split/<id>_compact.npy     ->  <base>/<id>_compact_split.npy

`<id>_sam.npy` and other stage files move to <base> if it has none yet. A SAM map identical to the one already
there is dropped; a different one (a separate SAM run) is kept next to it as <id>_sam_<tag>.npy and flagged in
the metadata. `<id>_results.json` is merged: saved_partitions and compact_files gain one entry per strategy.
Nothing is overwritten. Folders that a generator is writing to (*.tmp files) are skipped.

    python src/hpc/merge_compact_folders.py --root /beegfs/home/mrashid/datasets/SAM_BPT           # dry run
    python src/hpc/merge_compact_folders.py --root /beegfs/home/mrashid/datasets/SAM_BPT --apply
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import partition_files  # noqa: E402

SUFFIXED = re.compile(r'^(?P<base>.+_compact[a-z-]*_(?:\d+|singlepass))_(?P<tag>greedy|mso_[0-9.]+|split(?:_\d+)?)$')


def merge_metadata(target, source, tag, source_folder, sam_differs):
    """The target's metadata with this strategy's compact file added (or the source's, if the target has none)."""
    base = json.loads(target.read_text()) if target.is_file() else json.loads(source.read_text())
    incoming = json.loads(source.read_text())
    stage = partition_files.stage(tag)
    image_id = target.name.removesuffix('_results.json')
    saved = dict(base.get('saved_partitions') or {})
    saved.pop('compact', None) if not target.is_file() else None
    saved[stage] = str(target.with_name(f'{image_id}_{stage}.npy'))
    counts = dict(base.get('partition_label_counts') or {})
    if 'compact' in (incoming.get('partition_label_counts') or {}):
        counts[stage] = incoming['partition_label_counts']['compact']
        if not target.is_file():
            counts.pop('compact', None)
    strategy, alpha, gain = partition_files.parse_tag(tag)
    files = dict(base.get('compact_files') or {})
    files[tag] = {'strategy': strategy, 'mso_alpha': alpha, 'min_gain': gain, 'merged_from': source_folder,
                  'sam_run_differs': sam_differs}
    return {**base, 'saved_partitions': saved, 'partition_label_counts': counts, 'compact_files': files}


def merge(folder, root, apply, stats):
    match = SUFFIXED.match(folder.name)
    tag, target = match['tag'], root / match['base']
    if any(path.suffix == '.tmp' for path in folder.iterdir()):
        print(f'SKIP {folder.name}: *.tmp files present, a generator may still be writing (stop it first)')
        stats['skipped folders'] += 1
        return
    print(f'{folder.name} -> {target.name} (compact_{tag})')
    if apply:
        target.mkdir(exist_ok=True)
    stage = partition_files.stage(tag)
    ids = sorted(path.name.removesuffix('_results.json') for path in folder.glob('*_results.json'))
    for image_id in ids:
        compact = folder / f'{image_id}_compact.npy'
        if not compact.is_file():
            stats['images without compact file'] += 1
            continue
        destination = target / f'{image_id}_{stage}.npy'
        if destination.is_file():
            stats['compact already there (kept)'] += 1
            continue
        sam_differs = False
        sam, target_sam = folder / f'{image_id}_sam.npy', target / f'{image_id}_sam.npy'
        if sam.is_file():
            if not target_sam.is_file():
                apply and sam.rename(target_sam)
                stats['sam maps moved'] += 1
            elif np.array_equal(np.load(sam), np.load(target_sam)):
                apply and sam.unlink()
                stats['sam maps identical (dropped)'] += 1
            else:
                sam_differs = True
                apply and sam.rename(target / f'{image_id}_sam_{tag}.npy')
                stats['sam maps differ (kept as _sam_<tag>)'] += 1
        for other in folder.glob(f'{image_id}_*.npy'):
            if other.name not in {compact.name, sam.name} and not (target / other.name).exists():
                apply and other.rename(target / other.name)
        metadata = merge_metadata(target / f'{image_id}_results.json', folder / f'{image_id}_results.json',
                                  tag, folder.name, sam_differs)
        if apply:
            compact.rename(destination)
            (target / f'{image_id}_results.json').write_text(json.dumps(metadata, indent=2) + '\n')
            (folder / f'{image_id}_results.json').unlink()
        stats[f'compact_{tag} files'] += 1
    if apply and not any(folder.iterdir()):
        folder.rmdir()
        stats['folders removed'] += 1
    elif apply:
        print(f'  kept {folder.name}: {len(list(folder.iterdir()))} files left (see counts below)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='Partition root (output.partitions in the configs)')
    parser.add_argument('--apply', action='store_true', help='Move the files (default: only show what would happen)')
    args = parser.parse_args()
    folders = sorted(path for path in args.root.iterdir() if path.is_dir() and SUFFIXED.match(path.name))
    if not folders:
        print(f'No folders with a strategy suffix under {args.root}')
        return
    stats = Counter()
    for folder in folders:
        merge(folder, args.root, args.apply, stats)
    print()
    for key, value in sorted(stats.items()):
        print(f'{value:7d}  {key}')
    if not args.apply:
        print('\nDry run: nothing changed. Add --apply to merge.')


if __name__ == '__main__':
    main()
