"""Study manifests and completion checks must preserve inputs and unfinished work."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src/hpc' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_completion_requires_all_scientific_artifacts(tmp_path):
    driver = module('run_ablation_study')
    task = {'type': 'auc', 'output': str(tmp_path)}
    (tmp_path / 'comparison_status.json').write_text(json.dumps({'state': 'complete'}))
    assert not driver.task_complete(task)
    for name in ['summary.json', 'conclusion.md', 'measurements.csv', 'paired_comparisons.csv', 'auc_resolution_comparison.png']:
        (tmp_path / name).touch()
    assert driver.task_complete(task)
    (tmp_path / 'comparison_status.json').write_text(json.dumps({'state': 'running'}))
    assert not driver.task_complete(task)


def test_interim_target_does_not_complete_larger_study(tmp_path):
    driver = module('run_ablation_study')
    (tmp_path/'study.json').write_text(json.dumps({'state': 'interim', 'completed_images': ['a', 'b', 'c']}))
    for name in ['interim_conclusion.md', 'interim_selection.json', 'interim_comparisons.csv']:
        (tmp_path/name).touch()
    task = {'type': 'budget', 'output': str(tmp_path), 'completion_mode': 'interim', 'target_images': 3}
    assert driver.task_complete(task)
    assert not driver.task_complete(dict(task, target_images=6))
    assert not driver.task_complete({'type': 'budget', 'output': str(tmp_path)})


def test_preparation_freezes_common_pilot_before_execution(tmp_path):
    prepare = module('prepare_ablation_study')
    repo = tmp_path / 'repo'
    (repo / 'src/configs').mkdir(parents=True)
    partitions = tmp_path / 'partitions'
    variants = ['MSCOCO_sam1_vit_b_62_compact-sam_200', 'MSCOCO_sam2_large_62_compact-sam_200']
    for variant in variants:
        directory = partitions / variant
        directory.mkdir(parents=True)
        for number in range(1, 8):
            for suffix in ['_results.json', '_sam.npy', '_compact.npy']:
                (directory / f'{number:012d}{suffix}').write_bytes(b'fixture')
    config = {'data': {'name': 'MS_COCO', 'image_dir': '/frozen/images', 'annotation_file': '/frozen/coco.json'},
              'output': {'partitions': str(partitions)}}
    (repo / 'src/configs/MSCOCO_epito.yaml').write_text(yaml.safe_dump(config))
    home = tmp_path / 'study'
    with patch.object(sys, 'argv', ['prepare', '--repo', str(repo), '--study', str(home)]):
        prepare.main()
        before = (home / 'manifest.json').read_bytes()
        prepare.main()
    assert (home / 'manifest.json').read_bytes() == before
    manifest = json.loads(before)
    assert len(manifest['tasks']) == 4
    assert len(manifest['budget_protocol']['pilot_ids']) == 5
    assert manifest['budget_protocol']['eligible_common_images'] == 7
    for task in manifest['tasks']:
        assert '--image-ids' in task['args'] and '--reference-budget' in task['args']
        arguments = task['args']
        start = arguments.index('--image-ids') + 1
        assert arguments[start:start+5] == manifest['budget_protocol']['pilot_ids']


def test_full_budget_preparation_pairs_variants_and_freezes_statistics(tmp_path):
    prepare = module('prepare_full_budget_study')
    repo = tmp_path / 'repo'
    (repo / 'src/configs').mkdir(parents=True)
    partitions = tmp_path / 'partitions'
    image = tmp_path / 'input.jpg'
    image.write_bytes(b'input bytes to hash')
    for variant in ['ImageNet_sam1_vit_b_62_compact-sam_100', 'ImageNet_sam2_tiny_62_compact-sam_100']:
        folder = partitions / variant
        folder.mkdir(parents=True)
        for index in range(22):
            key = f'image{index:04d}'
            (folder / f'{key}_compact.npy').write_bytes(b'partition')
            (folder / f'{key}_results.json').write_text(json.dumps({
                'source_image': str(image), 'saved_partitions': {'compact': 'present'}}))
    (repo / 'src/configs/ImageNet_epito.yaml').write_text(yaml.safe_dump({
        'data': {'name': 'ImageNet'}, 'output': {'partitions': str(partitions)}}))
    manifest = prepare.prepare(repo, tmp_path / 'study', ['ImageNet'])
    assert len(manifest['tasks']) == 6
    assert manifest['budget_protocol']['background'] == 'full'
    assert manifest['budget_protocol']['margins']['relative_l1'] == .1
    ids = manifest['dataset_pools']['ImageNet']['image_ids']
    assert len(ids) == 20
    for task in manifest['tasks']:
        args = task['args']
        assert args[args.index('--bg-type')+1] == 'full'
        start = args.index('--image-ids')+1
        assert args[start:start+20] == ids
