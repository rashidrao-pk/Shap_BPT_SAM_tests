#!/usr/bin/env python3
"""Freeze a budget-only, full-background study before examining its outcomes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import yaml

MODELS = {
    'MSCOCO': [('OD_yolo', 'yolo11s'), ('OD_DETR', 'facebook/detr-resnet-50')],
    'ImageNet': [('OC_ResNet', 'resnet50'), ('OC_VGG', 'vgg16'), ('OC_ViT', 'vit_b_16')],
    'CelebAHQ': [('OC_CelebA', 'celeba_cnn')],
    'CAMO': [('COD_ViT', 'vit_b_16')],
    'BUSI': [('OC_BUSI', 'ghostsas001/busi-breast-ultrasound-siglip')],
    'PartImageNet': [('OC_PartImageNet', 'resnet50')],
    'FunnyBirds': [('OC_FunnyBirds', 'resnet50')],
    'PACO': [('OC_PACO', 'resnet50')],
    'KvasirSEG': [('OC_KvasirSEG', 'mmuratarat/kvasir-v2-classifier')],
    'ISIC2018': [('OC_ISIC2018', 'adeIaide/skin-lesion-7class-efficientnet')],
    'DOTA': [('OD_DOTA', 'yolo11s-obb')],
}
# Later datasets are opt-in through --datasets, so preparing the original study needs no new partitions.
DEFAULT_DATASETS = ['MSCOCO', 'ImageNet', 'CelebAHQ', 'CAMO', 'BUSI', 'PartImageNet']


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(repo, home, datasets, size=20):
    if (home / 'manifest.json').exists():
        raise ValueError('Study already frozen; use the existing manifest to resume')
    if size < 20:
        raise ValueError('Use at least 20 images per dataset for this statistical pilot')
    code = home / 'code'
    home.mkdir(parents=True, exist_ok=True)
    frozen = []
    for pattern in ['src/scripts/**/*.py', 'src/medical/*.py', 'src/configs/*.yaml',
                    'shap_bpt/*.py', 'shap_bpt/*.pyx', 'shap_bpt/*.so', 'src/imagenet_class_index.json']:
        for path in repo.glob(pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(repo)
            target = code / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            frozen.append({'path': str(relative), 'sha256': digest(target)})
    tasks, pools, inputs = [], {}, []
    for dataset in datasets:
        cfg = yaml.safe_load((repo / f'src/configs/{dataset}_epito.yaml').read_text())
        base = Path(cfg['output']['partitions'])
        variants = []
        completed = {}
        for sam in ['sam1', 'sam2']:
            choices = sorted(p for p in base.glob(f'{dataset}_{sam}_*compact-sam*') if p.is_dir())
            if len(choices) != 1:
                raise ValueError(f'{dataset}/{sam}: expected one partition folder, found {choices}')
            folder = choices[0]
            variants.append(folder)
            records = {}
            for path in folder.glob('*_results.json'):
                metadata = json.loads(path.read_text())
                image = Path(metadata.get('source_image', '/missing'))
                key = path.name.removesuffix('_results.json')
                stages = list(metadata.get('saved_partitions') or {}) or metadata.get('generation_settings', {}).get('save_partitions') or ['sam', 'refined']
                if image.is_file() and (folder / f'{key}_compact.npy').is_file() and all(
                        (folder / f'{key}_{stage}.npy').is_file() for stage in stages):
                    records[key] = image
            completed[folder.name] = records
        common = sorted(set.intersection(*(set(completed[p.name]) for p in variants)))
        if len(common) < size:
            raise ValueError(f'{dataset}: only {len(common)} eligible common images, need {size}')
        selected = sorted(random.Random(0).sample(common, size))
        pools[dataset] = {'eligible': len(common), 'image_ids': selected}
        for key in selected:
            images = {completed[p.name][key].resolve() for p in variants}
            if len(images) != 1:
                raise ValueError(f'{dataset}/{key}: SAM variants point to different source images')
            image = next(iter(images))
            inputs.append({'path': str(image), 'sha256': digest(image)})
        for folder in variants:
            config = json.loads(json.dumps(cfg))
            config['data']['mask_dir_final'] = str(folder)
            if config.get('model', {}).get('checkpoint'):
                config['model']['checkpoint'] = str((repo / config['model']['checkpoint']).resolve())
            config_path = home / 'configs' / f'{folder.name}.yaml'
            config_path.parent.mkdir(exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            for key in selected:
                for suffix in ['_compact.npy', '_results.json']:
                    path = folder / (key + suffix)
                    inputs.append({'path': str(path), 'sha256': digest(path)})
            for runner, model in MODELS[dataset]:
                root = home / 'budget'
                args = ['--runner', runner, '--model', model, '--config', str(config_path),
                    '--limit', str(size), '--image-ids', *selected, '--bg-type', 'full',
                    '--budgets', '100', '200', '500', '1000', '2000', '--reference-budget', '4000',
                    '--repeats', '3', '--pilot-seed', '0', '--holdout-fraction', '0.4',
                    '--auc-num-samples', '200', '--study-name', 'full_budget', '--device', 'cuda',
                    '--torch-threads', '6', '--output-dir', str(root)]
                output = root / runner / folder.name / 'bg_full'
                if runner == 'OD_yolo':
                    args += ['--checkpoint', str(repo / 'src/checkpoints/yolo11s.pt')]
                if runner == 'OC_BUSI':
                    args += ['--checkpoint', str(repo / 'checkpoints/busi_siglip')]
                if runner in {'OC_PartImageNet', 'OC_PACO'}:
                    args += ['--checkpoint', str(repo / 'checkpoints/partimagenet/resnet50-11ad3fa6.pth'),
                             '--weights', 'IMAGENET1K_V2']
                if runner == 'OC_CelebA':
                    args += ['--attribute', 'Brown_Hair']
                    output /= 'Brown_Hair'
                output /= 'full_budget'
                tasks.append({'id': f'budget_{runner}_{"sam1" if "_sam1_" in folder.name else "sam2"}',
                    'type': 'budget', 'script': 'src/scripts/explanations/budget_ablation.py',
                    'runner': runner, 'dataset': dataset, 'args': args, 'output': str(output)})
    protocol = {'background': 'full', 'backgrounds': ['black', 'gray', 'white', 'blurred', 'noise'],
        'budgets': [100, 200, 500, 1000, 2000], 'reference_budget': 4000, 'repeats': 3,
        'images_per_dataset': size, 'holdout_fraction': .4, 'seed': 0, 'auc_points_fixed': 200,
        'statistical_test': 'paired image-cluster bootstrap, simultaneous one-sided upper bounds',
        'bootstrap_resamples': 20000, 'alpha': .05,
        'margins': {'relative_l1': .10, 'spearman_loss': .05, 'top_jaccard_loss': .10,
                    'auc_ins_difference': .01, 'auc_del_difference': .01},
        'claim': 'Mean agreement with a 4000-budget approximation within each tested scope; not universal accuracy.',
        'rule': 'Select smallest calibration budget passing all scopes/metrics and all larger tested budgets; '
                'confirm once on held-out images without escalation; unstable references invalidate selection.'}
    manifest = {'study_root': str(home), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'repo': str(repo), 'scope': 'XAI explanation-budget only, full replacement',
        'postprocess': 'src/scripts/explanations/budget_noninferiority.py',
        'budget_protocol': protocol, 'dataset_pools': pools, 'frozen_inputs': inputs,
        'frozen_code': frozen, 'tasks': tasks}
    (home / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--datasets', nargs='+', choices=list(MODELS), default=DEFAULT_DATASETS)
    parser.add_argument('--images', type=int, default=20)
    args = parser.parse_args()
    result = prepare(args.repo.resolve(), args.study.resolve(), args.datasets, args.images)
    print(json.dumps({'tasks': len(result['tasks']), 'protocol': result['budget_protocol']}, indent=2))
