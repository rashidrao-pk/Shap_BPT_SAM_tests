"""PartImageNet split routing, COCO part masks, download validation, and XAI integration."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
sys.path.insert(0, str(ROOT / 'src/scripts/explanations'))
from partimagenet_data import collect_images
from ground_truth import load_ground_truth


def fixture(tmp_path):
    images = tmp_path / 'images'
    (images / 'val').mkdir(parents=True)
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    Image.fromarray(pixels).convert('RGB').save(images / 'val/n000001_1.JPEG')
    Image.fromarray(pixels).convert('RGB').save(images / 'val/unannotated.JPEG')
    annotation = tmp_path / 'val.json'
    annotation.write_text(json.dumps({
        'images': [{'id': 7, 'file_name': 'n000001_1.JPEG', 'height': 16, 'width': 16}],
        'categories': [{'id': 0, 'name': 'Quadruped Head'}, {'id': 1, 'name': 'Quadruped Body'}],
        'annotations': [
            {'id': 1, 'image_id': 7, 'category_id': 0, 'segmentation': [[1, 1, 6, 1, 6, 6, 1, 6]], 'area': 25, 'iscrowd': 0, 'bbox': [1, 1, 5, 5]},
            {'id': 2, 'image_id': 7, 'category_id': 1, 'segmentation': [[9, 9, 14, 9, 14, 14, 9, 14]], 'area': 25, 'iscrowd': 0, 'bbox': [9, 9, 5, 5]}]}))
    data = {'name': 'PartImageNet', 'input_format': 'directory', 'image_dir': str(images),
            'annotation_file': str(annotation), 'mask_dir_final': 'PartImageNet_test',
            'ground_truth': {'format': 'partimagenet', 'root': str(tmp_path), 'annotation_file': str(annotation)}}
    return data, images / 'val/n000001_1.JPEG'


def test_split_images_and_parts(tmp_path):
    data, image = fixture(tmp_path)
    records = collect_images(data)
    assert len(records) == 1 and records[0]['id'] == image.stem
    gt = load_ground_truth({'data': data}, image, (16, 16, 3))
    assert gt['mask'].sum() == 50 and gt['mask'][2, 2] and gt['mask'][10, 10]
    selected = load_ground_truth({'data': data}, image, (16, 16, 3), gt_class='Quadruped Head')
    assert selected['mask'].sum() == 25 and not selected['mask'][10, 10]
    assert selected['scope'] == 'selected_part'
    with pytest.raises(ValueError, match='Unknown PartImageNet part'):
        load_ground_truth({'data': data}, image, (16, 16, 3), gt_class='unknown')


def test_ambiguous_images_rejected(tmp_path):
    data, image = fixture(tmp_path)
    (tmp_path / 'images/other').mkdir()
    (tmp_path / 'images/other' / image.name).write_bytes(image.read_bytes())
    with pytest.raises(ValueError, match='expected one matching image'):
        collect_images(data)


def test_download_validation(tmp_path):
    import download_partimagenet as download
    bad = tmp_path / 'bad.zip'
    bad.write_text('<html>login</html>')
    with pytest.raises(ValueError, match='Not a ZIP'):
        download.extract(bad, tmp_path / 'output')
    with zipfile.ZipFile(bad, 'w') as archive:
        archive.writestr('../outside.png', b'bad')
    with pytest.raises(ValueError, match='Unsafe ZIP'):
        download.extract(bad, tmp_path / 'output')
    weights = tmp_path / 'weights.pth'
    weights.write_bytes(b'bad')
    with pytest.raises(ValueError, match='checksum mismatch'):
        download.check_hash(weights, '11ad3fa6')


def test_shared_explanations(tmp_path, monkeypatch):
    import yaml
    import experiment_common as common
    data, image = fixture(tmp_path)
    directory = tmp_path / 'partitions' / data['mask_dir_final']
    directory.mkdir(parents=True)
    partition = np.zeros((16, 16), dtype=np.uint16)
    partition[8:] = 1
    np.save(directory / f'{image.stem}_compact.npy', partition)
    (directory / f'{image.stem}_results.json').write_text(json.dumps({'saved_partitions': {'compact': 'path'}}))
    config = {'device': 'cpu', 'data': data,
              'output': {key: str(tmp_path / key) for key in ['partitions', 'explanations', 'evaluations']}}
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.safe_dump(config))
    class Adapter:
        class_names = ['object', 'other']
        def predict_batch(self, images):
            means = np.asarray([np.asarray(x).mean()/255 for x in images])
            return np.column_stack([means, 1-means])
        def detections(self, image):
            return []
    monkeypatch.setattr(common, 'load_adapter', lambda *args: Adapter())
    args = common.create_parser('OC_PartImageNet').parse_args([
        '--config', str(config_file), '--show-gt', '--budget', '8', '--no-auc',
        '--no-plots', '--no-html', '--torch-threads', '1'])
    common.run('OC_PartImageNet', args)
    assert len(list((tmp_path / 'explanations').rglob('*_complete.json'))) == 1
    gt_file = next((tmp_path / 'explanations').rglob('*_gt.json'))
    assert json.loads(gt_file.read_text())['foreground_pixels'] == 50


def test_dataset_download_cli_without_scientific_packages(tmp_path):
    import subprocess
    harness = '''
import importlib.abc, runpy, sys, urllib.request, zipfile
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'PIL', 'hf_xet', 'huggingface_hub', 'kagglehub'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
def retrieve(url, destination):
    with zipfile.ZipFile(destination, 'w') as archive:
        archive.writestr('PartImageNet_Seg/val/image.JPEG', b'fixture')
    return destination, None
urllib.request.urlretrieve = retrieve
script, root = sys.argv[1:]
sys.argv = [script, '--assets', 'dataset', '--dataset-root', root]
runpy.run_path(script, run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', harness,
                            str(ROOT / 'src/scripts/download_partimagenet.py'), str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'PartImageNet_Seg/val/image.JPEG').is_file()


def test_shared_generation_selects_annotation_split(tmp_path):
    from generate_coco_partitions import collect_images as collect
    data, image = fixture(tmp_path)
    assert [item['id'] for item in collect(data)] == [image.stem]


def test_configure_real_extracted_layout(tmp_path):
    import yaml
    import download_partimagenet as download
    data, image = fixture(tmp_path)
    config_path = tmp_path / 'config.yaml'
    old = {'data': {**data, 'image_dir': str(tmp_path / 'nonexistent/PartImageNet_Seg'),
                   'annotation_file': '/wrong/val.json'}, 'output': {'partitions': '/keep/partitions'}}
    config_path.write_text(yaml.safe_dump(old))
    records = download.configure_dataset(tmp_path, config_path)
    updated = yaml.safe_load(config_path.read_text())
    assert len(records) == 1
    assert updated['data']['image_dir'] == str(tmp_path)
    assert updated['data']['annotation_file'] == str(tmp_path / 'val.json')
    assert updated['data']['ground_truth']['annotation_file'] == updated['data']['annotation_file']
    assert updated['data']['mask_dir_final'] == data['mask_dir_final']
    assert updated['output'] == old['output']
    assert config_path.with_suffix('.yaml.bak').exists()


def test_configure_ambiguous_annotations_preserves_config(tmp_path):
    import download_partimagenet as download
    data, _ = fixture(tmp_path)
    (tmp_path / 'other').mkdir()
    (tmp_path / 'other/val.json').write_text((tmp_path / 'val.json').read_text())
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('keep this until selection is explicit')
    with pytest.raises(ValueError, match='found 2'):
        download.configure_dataset(tmp_path, config_path)
    assert config_path.read_text() == 'keep this until selection is explicit'


def test_configure_prefers_parts_over_whole_object_annotations(tmp_path):
    import yaml
    import download_partimagenet as download
    data, _ = fixture(tmp_path)
    annotation_text = (tmp_path / 'val.json').read_text()
    (tmp_path / 'val.json').unlink()
    for split in ['val', 'val_whole']:
        directory = tmp_path / 'PartImageNet/annotations' / split
        directory.mkdir(parents=True)
        (directory / 'val.json').write_text(annotation_text)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data': data, 'output': {'partitions': '/keep'}}))
    records = download.configure_dataset(tmp_path, config)
    selected = yaml.safe_load(config.read_text())['data']['annotation_file']
    assert Path(selected) == tmp_path / 'PartImageNet/annotations/val/val.json'
    assert len(records) == 1


def test_limited_partimagenet_previews_with_ground_truth(tmp_path, monkeypatch):
    import yaml
    from view_partitions import main as preview_main
    data, image = fixture(tmp_path)
    directory = tmp_path / 'partitions' / data['mask_dir_final']
    directory.mkdir(parents=True)
    labels = np.zeros((16, 16), dtype=np.uint16)
    labels[8:] = 1
    for stage in ['sam', 'compact']:
        np.save(directory / f'{image.stem}_{stage}.npy', labels)
    (directory / f'{image.stem}_results.json').write_text('{}')
    # Partial outputs must not be selected for preview.
    (directory / 'unannotated_results.json').write_text('{}')
    np.save(directory / 'unannotated_sam.npy', labels)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data': data,
        'output': {key: str(tmp_path / key) for key in ['partitions', 'evaluations', 'explanations']}}))
    monkeypatch.setattr(sys, 'argv', ['view_partitions.py', '--config', str(config),
        '--limit', '2', '--methods', 'sam', 'compact', '--show-gt'])
    preview_main()
    output = tmp_path / 'evaluations/previews' / data['mask_dir_final']
    assert (output / f'{image.stem}.png').is_file()
    assert not (output / 'unannotated.png').exists()
    saved_gt = json.loads((output / f'{image.stem}_gt.json').read_text())
    assert saved_gt['foreground_pixels'] == 50


def test_preview_uses_annotation_path_when_image_basename_is_duplicated(tmp_path, monkeypatch):
    import yaml
    from view_partitions import main as preview_main, resolve_paths
    from types import SimpleNamespace
    data, image = fixture(tmp_path)
    annotation = Path(data['annotation_file'])
    document = json.loads(annotation.read_text())
    document['images'][0]['file_name'] = 'val/' + image.name
    annotation.write_text(json.dumps(document))
    for split in ['train', 'test']:
        directory = tmp_path / 'images' / split
        directory.mkdir()
        Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)).save(directory / image.name)
    directory = tmp_path / 'partitions' / data['mask_dir_final']
    directory.mkdir(parents=True)
    for stage in ['sam', 'compact']:
        np.save(directory / f'{image.stem}_{stage}.npy', np.zeros((16, 16), dtype=np.uint16))
    (directory / f'{image.stem}_results.json').write_text('{}')
    # A completed output outside this annotation split must not enter the preview batch.
    for stage in ['sam', 'compact']:
        np.save(directory / f'unannotated_{stage}.npy', np.zeros((16, 16), dtype=np.uint16))
    (directory / 'unannotated_results.json').write_text('{}')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data': data, 'output': {
        key: str(tmp_path / key) for key in ['partitions', 'evaluations', 'explanations']}}))
    selected, _, _ = resolve_paths(SimpleNamespace(config=str(config), image_id=image.stem,
        image=None, partitions=None, output=None, methods=['sam', 'compact']))
    assert selected == image
    monkeypatch.setattr(sys, 'argv', ['view_partitions.py', '--config', str(config),
        '--limit', '5', '--methods', 'sam', 'compact', '--show-gt'])
    preview_main()
    output = tmp_path / 'evaluations/previews' / data['mask_dir_final']
    assert (output / f'{image.stem}.png').exists()
    assert not (output / 'unannotated.png').exists()
