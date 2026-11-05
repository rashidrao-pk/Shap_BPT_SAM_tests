#!/usr/bin/env python3
"""Freeze existing AUC inputs and select a common MS-COCO budget pilot before inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import zipfile
import yaml


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    repo, home = args.repo.resolve(), args.study.resolve()
    if (home / 'manifest.json').exists():
        print('Study already frozen:', home / 'manifest.json')
        return
    home.mkdir(parents=True, exist_ok=True)
    code = home / 'code'
    code_files = []
    paths = [*repo.glob('src/scripts/**/*.py'), *repo.glob('src/medical/*.py'),
             *repo.glob('src/configs/*.yaml'), *repo.glob('shap_bpt/*.py'),
             *repo.glob('shap_bpt/*.pyx'), *repo.glob('shap_bpt/*.so'), repo / 'src/imagenet_class_index.json']
    for path in paths:
        if not path.is_file():
            continue
        destination = code / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        code_files.append({'path': str(path.relative_to(repo)), 'sha256': digest(destination)})
    tasks, cases, excluded = [], [], []
    for info_path in sorted((repo / 'results/explanations').rglob('run_info.json')):
        info = json.loads(info_path.read_text())
        runner = info.get('runner') or ('OD_yolo' if info.get('model_group') == 'yolov11' else None)
        if runner is None:
            excluded.append({'path': str(info_path), 'reason': 'unknown runner'})
            continue
        eligible = []
        for archive in sorted(info_path.parent.glob('*/*_attributions.npz')):
            image_id = archive.name.removesuffix('_attributions.npz')
            prediction_file = archive.with_name(image_id + '_predictions.json')
            markers = [archive.with_name(image_id + suffix) for suffix in ['_complete.json', '_all_complete.json']]
            if not prediction_file.is_file() or not any(path.is_file() for path in markers):
                continue
            prediction = json.loads(prediction_file.read_text())
            image = prediction.get('source_image')
            if image is None and info.get('image_dir'):
                image = str(Path(info['image_dir']) / f"{int(prediction['image_id']):012d}.jpg")
            if image is None or not Path(image).is_file():
                continue
            with zipfile.ZipFile(archive) as z:
                methods = [name.removesuffix('.npy') for name in z.namelist() if name.endswith('.npy')]
            eligible.append((archive, prediction_file, Path(image), methods))
        if not eligible:
            excluded.append({'path': str(info_path), 'reason': 'no completed numeric maps with accessible source images'})
            continue
        selected = eligible[:3]  # Same predeclared first-N policy as compare_auc_points.py.
        relative = info_path.parent.relative_to(repo / 'results/explanations')
        snapshot = home / 'auc_inputs' / relative
        snapshot.mkdir(parents=True, exist_ok=True)
        shutil.copy2(info_path, snapshot / 'run_info.json')
        output = home / 'auc' / relative
        task_id = 'auc_' + runner + '_' + hashlib.sha256(str(relative).encode()).hexdigest()[:8]
        ids = []
        for archive, prediction_file, image, methods in selected:
            image_id = archive.name.removesuffix('_attributions.npz')
            destination = snapshot / archive.parent.name
            destination.mkdir(exist_ok=True)
            shutil.copy2(archive, destination / archive.name)
            shutil.copy2(prediction_file, destination / prediction_file.name)
            cases.append({'task_id': task_id, 'runner': runner, 'source_run': str(info_path.parent),
                'snapshot': str(snapshot), 'image_id': image_id, 'image': str(image),
                'background': info.get('background_replacement_values'), 'methods': methods,
                'archive_sha256': digest(archive), 'prediction_sha256': digest(prediction_file),
                'image_sha256': digest(image)})
            ids.append(image_id)
        tasks.append({'id': task_id, 'type': 'auc', 'script': 'src/scripts/explanations/compare_auc_points.py',
            'args': ['--run-dir', str(snapshot), '--image-ids', *ids, '--limit', str(len(ids)),
                     '--points', '100', '200', '--reference-points', '1000', '--repeats', '3',
                     '--tolerance', '0.005', '--device', 'cuda', '--torch-threads', '6', '--output-dir', str(output)],
            'output': str(output), 'source': str(info_path.parent), 'runner': runner})
    cfg = yaml.safe_load((repo / 'src/configs/MSCOCO_epito.yaml').read_text())
    base = Path(cfg['output']['partitions'])
    variants = ['MSCOCO_sam1_vit_b_62_compact-sam_200', 'MSCOCO_sam2_large_62_compact-sam_200']
    pools = []
    for name in variants:
        folder = base / name
        pool = set()
        for path in folder.glob('*_results.json'):
            image_id = path.name.removesuffix('_results.json')
            if all((folder / (image_id + '_' + stage + '.npy')).is_file() for stage in ['sam', 'compact']):
                pool.add(image_id)
        pools.append(pool)
    common = sorted(set.intersection(*pools))
    if len(common) < 5:
        raise ValueError('At least five common completed COCO partition inputs are required')
    pilot_ids = sorted(random.Random(0).sample(common, 5))
    pilot_inputs = []
    for variant in variants:
        config = json.loads(json.dumps(cfg))
        config['data']['mask_dir_final'] = str((base / variant).resolve())
        config_path = home / 'configs' / (variant + '.yaml')
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        for image_id in pilot_ids:
            for suffix in ['_compact.npy', '_sam.npy', '_results.json']:
                path = base / variant / (image_id + suffix)
                pilot_inputs.append({'path': str(path), 'sha256': digest(path)})
        for runner, model in [('OD_yolo', 'yolo11s'), ('OD_DETR', 'facebook/detr-resnet-50')]:
            output_root = home / 'budget'
            output = output_root / runner / variant / 'bg_black' / 'mscoco_pilot'
            arguments = ['--runner', runner, '--model', model, '--config', str(config_path),
                '--show-gt', '--limit', '5', '--image-ids', *pilot_ids,
                '--bg-type', 'black', '--class-aggregation', 'sum',
                '--budgets', '100', '200', '500', '1000', '2000', '--reference-budget', '4000',
                '--repeats', '3', '--pilot-seed', '0', '--holdout-fraction', '0.4',
                '--auc-num-samples', '200', '--study-name', 'mscoco_pilot', '--device', 'cuda',
                '--torch-threads', '6', '--output-dir', str(output_root)]
            if runner == 'OD_yolo':
                arguments += ['--checkpoint', str(repo / 'src/checkpoints/yolo11s.pt')]
            tasks.append({'id': 'budget_' + runner + '_' + ('sam1' if 'sam1' in variant else 'sam2'),
                'type': 'budget', 'script': 'src/scripts/explanations/budget_ablation.py',
                'args': arguments, 'output': str(output), 'runner': runner})
    manifest = {'study_root': str(home), 'created_utc': datetime.now(timezone.utc).isoformat(), 'repo': str(repo),
        'scope': 'AUC 100 vs 200 plus four MS-COCO explanation-budget pilots',
        'auc_protocol': {'points': [100, 200], 'reference_points': 1000, 'repeats': 3,
            'tolerance': .005, 'selection': 'First up to three completed IDs per saved run; all saved methods'},
        'budget_protocol': {'budgets': [100, 200, 500, 1000, 2000], 'reference_budget': 4000,
            'repeats': 3, 'auc_points': 200, 'eligible_common_images': len(common),
            'eligible_pool_selection_seed': 0, 'pilot_ids': pilot_ids, 'split_seed': 0,
            'holdout_fraction': .4, 'background': 'black'},
        'auc_cases': cases, 'pilot_partition_inputs': pilot_inputs,
        'excluded_runs': excluded, 'frozen_code': code_files, 'tasks': tasks}
    (home / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'manifest': str(home / 'manifest.json'), 'auc_tasks': sum(t['type']=='auc' for t in tasks),
        'budget_tasks': sum(t['type']=='budget' for t in tasks), 'pilot_ids': pilot_ids,
        'auc_image_cases': len(cases)}, indent=2))


if __name__ == '__main__':
    main()
