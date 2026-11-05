#!/usr/bin/env python3
"""Add all saved scalar AUC variants to existing experiment CSVs without inference."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from auc_export import enrich_auc_rows


def export(run_dir, evaluation_dir=None):
    targets = [run_dir / 'auc_results_all_images_current_run.csv']
    if evaluation_dir is not None:
        targets.append(evaluation_dir / targets[0].name)
    cache = {}
    for path in targets:
        if not path.is_file():
            continue
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = list(reader.fieldnames or []), list(reader)
        enriched = 0
        for row in rows:
            image_id = row.get('image_id') or row.get('image_no')
            if image_id not in cache:
                source = run_dir / image_id / f'{image_id}_metrics.json'
                cache[image_id] = json.loads(source.read_text()) if source.is_file() else []
            before = set(row)
            enrich_auc_rows([row], cache[image_id])
            enriched += bool(set(row) - before)
            fields.extend(key for key in row if key not in fields)
        backup = path.with_suffix(path.suffix + '.before_auc_variants.bak')
        if not backup.exists():
            shutil.copy2(path, backup)
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(path)
        print(f'{path}: {enriched}/{len(rows)} rows gained columns; backup: {backup.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--evaluation-dir', type=Path)
    args = parser.parse_args()
    export(args.run_dir, args.evaluation_dir)
