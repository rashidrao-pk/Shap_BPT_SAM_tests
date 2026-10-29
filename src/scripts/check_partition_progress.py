#!/usr/bin/env python3
"""Count completed partition images for all configured datasets, without SAM/GPU."""
import argparse
import json
import math
from pathlib import Path
import re

import yaml
from config_paths import output_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = ['MSCOCO_epito', 'ImageNet_epito', 'CelebAHQ_epito', 'CAMO_epito']
STAGES = ['boosted_filled', 'refined_filled', 'compact_selection', 'greedy_selection',
          'sam', 'refined', 'compact', 'boosted', 'filled', 'coverage']


def rooted(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(config_name):
    path = Path(config_name).expanduser()
    if not path.suffix:
        path = ROOT / 'src/configs' / f'{config_name}.yaml'
    else:
        path = rooted(path)
    return yaml.safe_load(path.read_text())


def load_data(config_name):
    return load_config(config_name)['data']


def source_ids(data, limit=None):
    if data.get('input_format', 'coco') == 'directory':
        folder = rooted(data['image_dir'])
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        ids, seen = [], set()
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if path.stem in seen:
                raise ValueError(f'Duplicate input image stem: {path.stem}')
            ids.append(path.stem)
            seen.add(path.stem)
    else:
        records = json.loads(rooted(data['annotation_file']).read_text())['images']
        ids = [f"{int(record['id']):012d}" for record in sorted(records, key=lambda r: int(r['id']))]
    return set(ids if limit is None else ids[:limit])


def format_bytes(size):
    value = float(size)
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024


def format_duration(seconds, samples):
    if not samples:
        return 'n/a'
    seconds = round(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def output_image_id(name):
    for suffix in ['_results.json', '_raw_masks.npz', *[f'_{stage}.npy' for stage in STAGES]]:
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return None


def inspect_folder(folder, expected_ids, dataset, methods=None):
    files = {path.name for path in folder.iterdir() if path.is_file()} if folder.is_dir() else set()
    complete, observed = set(), set()
    invalid_metadata = 0
    subset_bytes = folder_bytes = 0
    subset_seconds = folder_seconds = 0.0
    subset_timed = folder_timed = folder_completed = 0
    if folder.is_dir():
        for path in folder.rglob('*'):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue  # An active generator may atomically rename a temporary file.
            folder_bytes += size
            if path.parent == folder and output_image_id(path.name) in expected_ids:
                subset_bytes += size
    for name in files:
        if name.endswith('_results.json'):
            image_id = name.removesuffix('_results.json')
            in_subset = image_id in expected_ids
            if in_subset:
                observed.add(image_id)
            try:
                record = json.loads((folder / name).read_text())
                if not isinstance(record, dict):
                    raise ValueError('Expected metadata object')
                stages = methods or list(record.get('saved_partitions') or {})
                if not stages:
                    stages = record.get('generation_settings', {}).get('save_partitions') or ['sam', 'refined']
                if not isinstance(stages, list) or not all(isinstance(stage, str) for stage in stages):
                    raise ValueError('Invalid saved stages')
                if all(f'{image_id}_{stage}.npy' in files for stage in stages):
                    folder_completed += 1
                    if in_subset:
                        complete.add(image_id)
                    seconds = record.get('total_time_sec')
                    if (isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                            and math.isfinite(seconds) and seconds >= 0):
                        folder_seconds += seconds
                        folder_timed += 1
                        if in_subset:
                            subset_seconds += seconds
                            subset_timed += 1
            except (OSError, ValueError, TypeError, AttributeError):
                if in_subset:
                    invalid_metadata += 1
        elif name.endswith('.npy'):
            for stage in STAGES:
                suffix = f'_{stage}.npy'
                if name.endswith(suffix):
                    image_id = name.removesuffix(suffix)
                    if image_id in expected_ids:
                        observed.add(image_id)
                    break
    total = len(expected_ids)
    return {'dataset': dataset, 'folder': str(folder), 'completed': len(complete),
            'total': total, 'percent': round(100 * len(complete) / total, 2) if total else 0,
            'partial': len(observed - complete), 'remaining': total - len(complete),
            'invalid_metadata': invalid_metadata,
            'subset_bytes': subset_bytes, 'folder_bytes': folder_bytes,
            'subset_generation_seconds': subset_seconds, 'folder_generation_seconds': folder_seconds,
            'subset_timed_images': subset_timed, 'folder_timed_images': folder_timed,
            'subset_missing_timings': len(complete) - subset_timed,
            'folder_completed_images': folder_completed,
            'folder_missing_timings': folder_completed - folder_timed,
            'status': 'present' if folder.is_dir() else 'not created'}


def check_dataset(config_name, methods=None, limit=None, configured_only=False):
    config = load_config(config_name)
    data = config['data']
    expected = source_ids(data, limit)
    base = output_path(config, 'partitions')
    configured = base / data['mask_dir_final']
    folders = {configured}
    if not configured_only and base.is_dir():
        prefix = re.sub(r'[^A-Za-z0-9-]', '', data['name']) + '_sam'
        folders.update(path for path in base.iterdir() if path.is_dir() and path.name.startswith(prefix))
    return [inspect_folder(folder, expected, data['name'], methods) for folder in sorted(folders)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', '--configs', nargs='+', default=DEFAULT_CONFIGS)
    parser.add_argument('--methods', nargs='+', choices=STAGES,
                        help='Require these stages; default: stages recorded for each image')
    parser.add_argument('--configured-only', action='store_true', help='Check only data.mask_dir_final')
    parser.add_argument('--limit', type=int, help='Compare against first N inputs in generation order')
    parser.add_argument('--json', type=Path, help='Also write a JSON report')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    rows, errors = [], []
    for config in args.config:
        try:
            rows.extend(check_dataset(config, args.methods, args.limit, args.configured_only))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'config': config, 'error': str(exc)})
    print(f'{"Dataset":<12} {"Completed / Total":>19} {"Progress":>9} {"Partial":>8} '
          f'{"Remaining":>10} {"Size(subset)":>12} {"Size(folder)":>12} '
          f'{"Time(subset)":>12} {"Time(folder)":>12}  Folder')
    for row in rows:
        counts = f'{row["completed"]} / {row["total"]}'
        print(f'{row["dataset"]:<12} {counts:>19} {row["percent"]:>8.2f}% '
              f'{row["partial"]:>8} {row["remaining"]:>10} '
              f'{format_bytes(row["subset_bytes"]):>12} {format_bytes(row["folder_bytes"]):>12} '
              f'{format_duration(row["subset_generation_seconds"], row["subset_timed_images"]):>12} '
              f'{format_duration(row["folder_generation_seconds"], row["folder_timed_images"]):>12}  '
              f'{Path(row["folder"]).name}')
        if row['status'] != 'present' or row['invalid_metadata']:
            print(f'  {row["status"]}; invalid metadata: {row["invalid_metadata"]}')
        if row['subset_missing_timings'] or row['folder_missing_timings']:
            print(f'  Completed images missing timings: subset={row["subset_missing_timings"]}, '
                  f'folder={row["folder_missing_timings"]}')
    print('Size = logical file sizes; subset follows --limit, folder includes all files. '
          'Time = sum of recorded per-image generation times (HH:MM:SS), not Slurm wall time.')
    for error in errors:
        print(f'ERROR {error["config"]}: {error["error"]}')
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({'datasets': rows, 'errors': errors}, indent=2) + '\n')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
