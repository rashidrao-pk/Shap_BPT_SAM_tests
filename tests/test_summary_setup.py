"""Setup summary: checkpoint placement, ground-truth coverage, readiness, and failure isolation."""
import json
from pathlib import Path
import sys

from PIL import Image
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import summary_setup as setup
from check_partition_progress import DEFAULT_CONFIGS


def write_config(path, data, partitions, model=None):
    config = {'data': data, 'output': {'partitions': str(partitions), 'explanations': str(path.parent / 'e'),
                                       'evaluations': str(path.parent / 'v')}}
    if model:
        config['model'] = {'checkpoint': str(model)}
    path.write_text(yaml.safe_dump(config))
    return str(path)


def kvasir(tmp_path, masks=2):
    images, gt = tmp_path / 'images', tmp_path / 'masks'
    images.mkdir()
    gt.mkdir()
    for index in range(2):
        Image.new('RGB', (8, 8)).save(images / f'a{index}.jpg')
    for index in range(masks):
        Image.new('L', (8, 8)).save(gt / f'a{index}.jpg')
    data = {'name': 'KvasirSEG', 'input_format': 'directory', 'image_dir': str(images),
            'mask_dir_final': 'KvasirSEG_sam2_tiny_62_compact-sam_50',
            'ground_truth': {'format': 'kvasir_seg', 'root': str(gt)}}
    checkpoint = tmp_path / 'kvasir_vit'
    checkpoint.mkdir()
    for name in ['config.json', 'pytorch_model.bin']:
        (checkpoint / name).write_text('x')
    return data, checkpoint


def partitions(root, name, ids):
    folder = root / name
    folder.mkdir(parents=True)
    for image_id in ids:
        (folder / f'{image_id}_compact.npy').write_bytes(b'x')
        (folder / f'{image_id}_results.json').write_text('{}')


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(setup, 'package_status', lambda name: True)
    monkeypatch.setattr(setup, 'shap_bpt_extension', lambda: setup.Check(setup.OK, 'built', ''))
    monkeypatch.setattr(setup, 'check_model', wrap_sam(setup.check_model))


def wrap_sam(original):
    def check(model, override=None, pins=None):
        if model.label.startswith('SAM'):
            return setup.ModelResult(model, setup.OK, '/sam', 1, '')
        return original(model, override, pins)
    return check


def test_every_default_config_is_registered():
    for name in DEFAULT_CONFIGS:
        data = yaml.safe_load((ROOT / f'src/configs/{name}.yaml').read_text())['data']
        assert data['name'].replace('_', '').lower() in setup.DATASETS, name


def test_truncated_checkpoint_is_not_ok(tmp_path):
    path = tmp_path / 'model.pt'
    path.write_bytes(b'abc')
    model = setup.Model('Test', 'X.md', 'file', (str(path),), size=10)
    result = setup.check_model(model)
    assert result.status == setup.MISSING and 'incomplete' in result.detail
    path.write_bytes(b'0123456789')
    assert setup.check_model(model).status == setup.OK


def test_folder_checkpoint_needs_every_file(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    model = setup.Model('Test', 'X.md', 'folder', (str(tmp_path),), files=('config.json', 'model.safetensors'))
    result = setup.check_model(model)
    assert result.status == setup.MISSING and 'model.safetensors' in result.detail


def test_config_checkpoint_overrides_the_default(tmp_path):
    path = tmp_path / 'elsewhere.pth'
    path.write_bytes(b'abc')
    model = setup.Model('Test', 'X.md', 'file', ('checkpoints/absent.pth',))
    assert setup.check_model(model, override=path).location == str(path)


def test_sam2_searches_the_environment_folder(tmp_path, monkeypatch):
    folder = tmp_path / 'checkpoints'
    folder.mkdir()
    (folder / 'sam2.1_hiera_tiny.pt').write_bytes(b'x' * 4)
    monkeypatch.setenv('SAM2_DIR', str(tmp_path))
    model = setup.Model('SAM-2 Tiny', 'SAM2.md', 'file', ('sam2.1_hiera_tiny.pt',), size=4)
    assert setup.check_model(model).status == setup.OK


def test_coco_uses_the_large_sam2_checkpoint():
    assert setup.DATASETS['mscoco'].sam2 == 'sam2_large'
    assert setup.DATASETS['camo'].sam2 == 'sam2_tiny'


def test_hf_cache_without_weights_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path))
    snapshot = tmp_path / 'models--facebook--detr-resnet-50/snapshots/abc'
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    assert setup.check_model(setup.MODELS['detr']).status == setup.MISSING
    (snapshot / 'model.safetensors').write_bytes(b'weights')
    assert setup.check_model(setup.MODELS['detr']).status == setup.OK


def test_torch_cache_honours_torch_home(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_HOME', str(tmp_path))
    assert setup.check_model(setup.MODELS['vit_b16']).location.startswith(str(tmp_path))


def test_hash_mismatch_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, 'candidates', lambda model: [tmp_path / 'x'])
    tampered = tmp_path / 'x'
    tampered.write_bytes(b'abc')
    monkeypatch.setattr(setup, 'verify_hash', lambda found, pins: 'SHA-256 differs from the pinned value')
    result = setup.check_model(setup.Model('Test', 'X.md', 'file', ('x',)), pins={'x': '0'})
    assert result.status == setup.MISSING and 'SHA-256' in result.detail


def test_pinned_hashes_cover_the_download_scripts():
    pins = setup.pinned_hashes()
    assert len(pins['checkpoints/funnybirds/resnet50_final_0_checkpoint_best.pth.tar']) == 64
    assert 'src/checkpoints/celeba_model.pth' in pins


def test_partial_ground_truth_warns_and_names_the_gap(tmp_path):
    data, _ = kvasir(tmp_path, masks=1)
    check = setup.check_gt(setup.DATASETS['kvasirseg'], data, {'a0', 'a1'})
    assert check.status == setup.WARN and check.text == '1/2' and '1 images' in check.detail


def test_celeba_mask_folders_are_counted_by_image(tmp_path):
    folder = tmp_path / '0'
    folder.mkdir()
    Image.new('L', (4, 4)).save(folder / '00000_hair.png')
    Image.new('L', (4, 4)).save(folder / '00000_nose.png')
    data = {'ground_truth': {'format': 'celebamask_hq', 'root': str(tmp_path)}}
    check = setup.check_gt(setup.DATASETS['celebahq'], data, {'0', '2000'})
    assert check.status == setup.WARN and check.text == '1/2'


def test_imagenet_subset_is_expected_for_the_default_variant(tmp_path):
    spec = setup.Dataset('ImageNet', 'x.md', gt=2)
    folder = tmp_path / 'ImageNetS50/validation-segmentation/n01443537'
    folder.mkdir(parents=True)
    for name in ['a', 'b']:
        Image.new('RGB', (4, 4)).save(folder / f'{name}.png')
    data = {'ground_truth': {'format': 'imagenets', 'root': str(tmp_path), 'variant': 50}}
    assert setup.check_gt(spec, data, {'a', 'b', 'c', 'd'}).status == setup.OK


def test_paco_split_mismatch_is_reported(tmp_path):
    data = {'annotation_file': str(tmp_path / 'val.json'),
            'ground_truth': {'format': 'paco', 'root': str(tmp_path), 'annotation_file': str(tmp_path / 'test.json')}}
    check = setup.check_gt(setup.DATASETS['paco'], data, {'a'})
    assert check.status == setup.MISSING and 'different splits' in check.detail


def test_partition_count_needs_compact_and_metadata(tmp_path):
    partitions(tmp_path, 'done', ['a'])
    (tmp_path / 'done/b_compact.npy').write_bytes(b'x')
    assert setup.completed_partitions(tmp_path / 'done', {'a', 'b', 'c'}) == 1


def test_no_partitions_is_a_next_step_not_a_failure(tmp_path, installed):
    data, checkpoint = kvasir(tmp_path)
    config = write_config(tmp_path / 'c.yaml', data, tmp_path / 'parts', checkpoint)
    results, _, _ = setup.build_report([config])
    assert (results[0]['can_partition'], results[0]['can_explain']) == ('READY', 'NO PARTS')


def test_complete_dataset_is_ready_for_xai(tmp_path, installed):
    data, checkpoint = kvasir(tmp_path)
    partitions(tmp_path / 'parts', data['mask_dir_final'], ['a0', 'a1'])
    config = write_config(tmp_path / 'c.yaml', data, tmp_path / 'parts', checkpoint)
    result = setup.build_report([config])[0][0]
    assert result['can_explain'] == 'READY' and result['partitions']['sam2'].text == '2/2'


def test_folder_other_than_the_configured_one_is_not_ready(tmp_path, installed):
    data, checkpoint = kvasir(tmp_path)
    partitions(tmp_path / 'parts', 'KvasirSEG_sam1_vit_b_62_compact-sam_50', ['a0', 'a1'])
    config = write_config(tmp_path / 'c.yaml', data, tmp_path / 'parts', checkpoint)
    result = setup.build_report([config])[0][0]
    assert result['partitions']['sam1'].text == '2/2' and result['can_explain'] == 'NO PARTS'


def test_missing_model_blocks_xai_only(tmp_path, installed):
    data, checkpoint = kvasir(tmp_path)
    (checkpoint / 'pytorch_model.bin').unlink()
    config = write_config(tmp_path / 'c.yaml', data, tmp_path / 'parts', checkpoint)
    result = setup.build_report([config])[0][0]
    assert (result['can_partition'], result['can_explain']) == ('READY', 'BLOCKED')


def test_broken_config_does_not_stop_other_datasets(tmp_path, installed, capsys):
    data, checkpoint = kvasir(tmp_path)
    good = write_config(tmp_path / 'c.yaml', data, tmp_path / 'parts', checkpoint)
    bad = tmp_path / 'bad.yaml'
    bad.write_text(yaml.safe_dump({'data': {'name': 'Unknown'}}))
    with pytest.raises(SystemExit) as stopped:
        setup.main(['--config', str(bad), good])
    assert stopped.value.code == 1
    output = capsys.readouterr().out
    assert 'Kvasir-SEG' in output and 'bad config' in output and 'To fix' in output


def test_main_writes_json_and_strict_requires_xai(tmp_path, installed, capsys):
    data, checkpoint = kvasir(tmp_path)
    config = write_config(tmp_path / 'c.yaml', data, tmp_path / 'parts', checkpoint)
    report = tmp_path / 'report/setup.json'
    setup.main(['--config', config, '--json', str(report)])
    assert json.loads(report.read_text())['datasets'][0]['can_explain'] == 'NO PARTS'
    with pytest.raises(SystemExit):
        setup.main(['--config', config, '--strict'])


def test_login_node_hint_accompanies_missing_packages():
    lines = setup.problem_lines([], {'torch': False}, setup.Check(setup.OK, 'built', ''))
    assert any('GPU node' in line for line in lines)


def test_partimagenet_unannotated_val_image_is_expected(tmp_path):
    annotation = tmp_path / 'val.json'
    annotation.write_text(json.dumps({'images': [{'id': 1, 'file_name': 'a.JPEG'}, {'id': 2, 'file_name': 'b.JPEG'}],
                                      'annotations': [{'image_id': 1}]}))
    data = {'annotation_file': str(annotation),
            'ground_truth': {'format': 'partimagenet', 'root': str(tmp_path), 'annotation_file': str(annotation)}}
    assert setup.check_gt(setup.Dataset('Parts', 'x.md', gt=1), data, {'a', 'b'}).status == setup.OK


def test_table_aligns_columns():
    lines = setup.table(['A', 'Long'], [['value', 1]]).splitlines()
    assert lines[0].index('Long') == lines[2].index('1')
