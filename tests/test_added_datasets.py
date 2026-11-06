"""FunnyBirds, PACO, Kvasir-SEG and ISIC 2018: GT decoding, adapters, downloads, and registration."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
sys.path.insert(0, str(ROOT / 'src/scripts/explanations'))
sys.path.insert(0, str(ROOT / 'src/hpc'))
from ground_truth import load_ground_truth, semantic_styles
from partimagenet_data import collect_images

RUNNERS = {'OC_FunnyBirds': 'FunnyBirds', 'OC_PACO': 'PACO', 'OC_KvasirSEG': 'KvasirSEG', 'OC_ISIC2018': 'ISIC2018'}


def funnybirds(tmp_path):
    image = tmp_path / 'FunnyBirds/test/3/000030.png'
    part_map = tmp_path / 'FunnyBirds/test_part_map/3/000030.png'
    image.parent.mkdir(parents=True)
    part_map.parent.mkdir(parents=True)
    Image.new('RGBA', (8, 8), (9, 9, 9, 255)).save(image)
    colors = np.zeros((8, 8, 4), dtype=np.uint8)
    colors[..., 3] = 255
    colors[0, :2, :3] = (255, 255, 253), (255, 255, 254)  # Two eyes, one semantic part.
    colors[1, 0, :3] = 255, 255, 0                        # Beak.
    colors[2, :3, :3] = 170, 170, 170                     # Body.
    colors[3, :4, :3] = 204, 204, 207                     # Background distractor.
    colors[4, 0, :3] = 0, 128, 1                          # Anti-aliased wing edge.
    Image.fromarray(colors).save(part_map)
    settings = {'format': 'funnybirds', 'root': str(tmp_path / 'FunnyBirds'), 'split': 'test'}
    return {'data': {'name': 'FunnyBirds', 'ground_truth': settings}}, image


def test_funnybirds_default_gt_is_only_class_defining_parts(tmp_path):
    config, image = funnybirds(tmp_path)
    gt = load_ground_truth(config, image, (8, 8, 3))
    assert gt['foreground_pixels'] == 3  # Excludes body, distractors, and edge colours.
    assert gt['part_names'] == ['beak', 'eye'] and gt['dataset_class'] == '3'
    assert {style['name'] for style in semantic_styles(gt)} == {'beak', 'eye'}
    json.dumps({key: value for key, value in gt.items() if key not in {'mask', 'label_map'}})


def test_funnybirds_part_selection(tmp_path):
    config, image = funnybirds(tmp_path)
    assert load_ground_truth(config, image, (8, 8, 3), gt_class='eye')['foreground_pixels'] == 2
    assert load_ground_truth(config, image, (8, 8, 3), gt_class='body')['foreground_pixels'] == 3
    hidden = load_ground_truth(config, image, (8, 8, 3), gt_class='tail')
    assert hidden['foreground_pixels'] == 0 and hidden['status'] == 'selected part not visible'
    with pytest.raises(ValueError, match='Unknown FunnyBirds part'):
        load_ground_truth(config, image, (8, 8, 3), gt_class='fin')


def test_funnybirds_missing_part_map_is_reported(tmp_path):
    config, image = funnybirds(tmp_path)
    config['data']['ground_truth']['split'] = 'train'
    assert load_ground_truth(config, image, (8, 8, 3))['status'] == 'annotation not available'


def paco(tmp_path):
    images = tmp_path / 'images/train2017'
    images.mkdir(parents=True)
    Image.new('RGB', (16, 16)).save(images / '000000000007.jpg')
    Image.new('RGB', (16, 16)).save(images / '000000000008.jpg')  # Not in this split.
    annotation = tmp_path / 'paco_val.json'

    def box(identifier, category, left, right):
        return {'id': identifier, 'image_id': 7, 'category_id': category, 'iscrowd': 0, 'area': 1,
                'bbox': [left, left, right - left, right - left],
                'segmentation': [[left, left, right, left, right, right, left, right]]}
    annotation.write_text(json.dumps({
        'images': [{'id': 7, 'file_name': 'train2017/000000000007.jpg', 'height': 16, 'width': 16}],
        'categories': [{'id': 1, 'name': 'mug', 'supercategory': 'OBJECT'},
                       {'id': 2, 'name': 'mug:handle', 'supercategory': 'PART'},
                       {'id': 3, 'name': 'mug:rim', 'supercategory': 'PART'}],
        'annotations': [box(1, 1, 0, 15), box(2, 2, 1, 4), box(3, 3, 8, 11)]}))
    settings = {'format': 'paco', 'root': str(tmp_path), 'annotation_file': str(annotation)}
    data = {'name': 'PACO', 'image_dir': str(tmp_path / 'images'), 'annotation_file': str(annotation),
            'ground_truth': settings}
    return data, images / '000000000007.jpg'


def test_paco_lists_split_images_with_their_subfolder(tmp_path):
    data, _ = paco(tmp_path)
    assert [(item['id'], item['file_name']) for item in collect_images(data)] == [
        ('000000000007', 'train2017/000000000007.jpg')]


def test_paco_default_gt_excludes_whole_objects(tmp_path):
    data, image = paco(tmp_path)
    parts = load_ground_truth({'data': data}, image, (16, 16, 3))
    assert sorted(parts['part_names']) == ['mug:handle', 'mug:rim']
    assert not parts['mask'][14, 14]
    whole = load_ground_truth({'data': data}, image, (16, 16, 3), gt_class='mug')
    assert whole['mask'][14, 14] and whole['foreground_pixels'] > parts['foreground_pixels']
    handle = load_ground_truth({'data': data}, image, (16, 16, 3), gt_class='MUG:handle')
    assert handle['part_names'] == ['mug:handle'] and not handle['mask'][9, 9]
    with pytest.raises(ValueError, match='Unknown PACO part'):
        load_ground_truth({'data': data}, image, (16, 16, 3), gt_class='mug:lid')


@pytest.mark.parametrize('format_name,name', [('kvasir_seg', 'abc.jpg'), ('isic2018', 'abc_segmentation.png')])
def test_lesion_masks_match_by_stem(tmp_path, format_name, name):
    pixels = np.zeros((8, 8), dtype=np.uint8)
    pixels[:2] = 255
    Image.fromarray(pixels).save(tmp_path / name, quality=100)
    config = {'data': {'ground_truth': {'format': format_name, 'root': str(tmp_path)}}}
    gt = load_ground_truth(config, 'images/abc.jpg', (8, 8, 3))
    assert gt['foreground_pixels'] == 16 and gt['scope'] == 'lesion_region'
    assert load_ground_truth(config, 'images/other.jpg', (8, 8, 3))['status'] == 'annotation not available'
    with pytest.raises(ValueError, match='binary'):
        load_ground_truth(config, 'images/abc.jpg', (8, 8, 3), gt_class='polyp')


def test_kvasir_jpeg_noise_does_not_enter_the_mask(tmp_path):
    pixels = np.zeros((8, 8), dtype=np.uint8)
    pixels[0, 0], pixels[1, 1], pixels[2, 2] = 255, 127, 3
    Image.fromarray(pixels).convert('RGB').save(tmp_path / 'abc.png')
    (tmp_path / 'abc.png').rename(tmp_path / 'abc.jpg')  # The loader reads the content, not the extension.
    config = {'data': {'ground_truth': {'format': 'kvasir_seg', 'root': str(tmp_path)}}}
    assert load_ground_truth(config, 'abc.jpg', (8, 8, 3))['foreground_pixels'] == 1


def adapter_args(checkpoint, **values):
    return argparse.Namespace(checkpoint=checkpoint, model_batch_size=2, local_files_only=True,
                              model=None, **values)


def test_funnybirds_adapter_skips_imagenet_normalization(tmp_path):
    import torch
    from torchvision.models import resnet50
    from dataset_adapters import FunnyBirdsAdapter
    path = tmp_path / 'checkpoint.pth.tar'
    torch.save({'state_dict': resnet50(weights=None, num_classes=50).state_dict(), 'epoch': 1}, path)
    adapter = FunnyBirdsAdapter(adapter_args(path), 'cpu')
    tensor = adapter.preprocess(Image.new('RGBA', (32, 32), (255, 0, 0, 0)))
    assert tensor.shape == (3, 256, 256) and float(tensor.max()) == 1 and float(tensor.min()) == 0
    assert adapter.name == 'resnet50' and adapter.preprocessing_geometry == 'resize'
    scores = adapter.predict_batch(np.zeros((3, 32, 32, 3), dtype=np.uint8))
    assert scores.shape == (3, 50) and np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())


def test_funnybirds_adapter_rejects_imagenet_head(tmp_path):
    import torch
    from torchvision.models import resnet50
    from dataset_adapters import FunnyBirdsAdapter
    path = tmp_path / 'imagenet.pth'
    torch.save({'state_dict': resnet50(weights=None).state_dict()}, path)
    with pytest.raises(ValueError, match='50-class'):
        FunnyBirdsAdapter(adapter_args(path), 'cpu')


def test_isic_adapter_uses_checkpoint_label_order_and_gradcam_layer(tmp_path):
    safetensors = pytest.importorskip('safetensors.torch')
    import torch
    from dataset_adapters import IsicAdapter, efficientnet_b0
    names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    (tmp_path / 'config.json').write_text(json.dumps({
        'class_names': names, 'image_size': 64, 'dropout': .3,
        'normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]}}))
    safetensors.save_file(efficientnet_b0(7, .3).state_dict(), tmp_path / 'model.safetensors')
    adapter = IsicAdapter(adapter_args(tmp_path), 'cpu')
    assert adapter.class_names == names and not adapter.model.training
    assert adapter.preprocess(Image.new('RGB', (100, 50))).shape == (3, 64, 64)  # Direct, not center-cropped.
    assert isinstance(adapter.model.encoder[0][-1], torch.nn.Sequential)
    assert adapter.predict_batch(np.zeros((1, 80, 80, 3), dtype=np.uint8)).shape == (1, 7)


def test_kvasir_adapter_reads_labels_and_rejects_other_models(tmp_path):
    transformers = pytest.importorskip('transformers')
    from dataset_adapters import KvasirAdapter
    labels = {0: 'esophagitis', 1: 'polyps'}
    config = transformers.ViTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                    intermediate_size=16, image_size=224, id2label=labels,
                                    label2id={value: key for key, value in labels.items()})
    transformers.ViTForImageClassification(config).save_pretrained(tmp_path)
    adapter = KvasirAdapter(adapter_args(tmp_path), 'cpu')
    assert adapter.class_names == ['esophagitis', 'polyps'] and adapter.name == 'vit_kvasir'
    from all_methods import vit_gradcam_layer
    assert vit_gradcam_layer(adapter.model, adapter.name) is not None
    assert adapter.predict_batch(np.zeros((1, 40, 40, 3), dtype=np.uint8)).shape == (1, 2)
    from all_methods import GradientMethods
    from experiment_common import create_parser
    image = np.full((40, 40, 3), 100, dtype=np.uint8)
    engine = GradientMethods(adapter, image, [np.zeros_like(image)],
                             create_parser('OC_KvasirSEG').parse_args([]))
    attribution = engine.compute('GradCAM', 1)
    assert attribution.shape == (1, 40, 40) and np.isfinite(attribution).all()
    assert not vit_gradcam_layer(adapter.model, adapter.name)._forward_hooks
    config.id2label = {0: 'cat', 1: 'dog'}
    transformers.ViTForImageClassification(config).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match='Kvasir v2'):
        KvasirAdapter(adapter_args(tmp_path), 'cpu')


def test_isic_preparation_keeps_masks_binary_and_aligned(tmp_path):
    from download_datasets import prepare_isic
    images, masks = tmp_path / 'images', tmp_path / 'masks'
    images.mkdir()
    masks.mkdir()
    Image.new('RGB', (400, 200), (10, 20, 30)).save(images / 'ISIC_0000001.jpg')
    pixels = np.zeros((200, 400), dtype=np.uint8)
    pixels[:, :200] = 255
    Image.fromarray(pixels).save(masks / 'ISIC_0000001_segmentation.png')
    (images / 'ATTRIBUTION.txt').write_text('not an image')
    output = tmp_path / 'prepared'
    assert prepare_isic(images, masks, output, 100) == 1
    image = Image.open(output / 'images/ISIC_0000001.png')
    mask = np.asarray(Image.open(output / 'masks/ISIC_0000001_segmentation.png'))
    assert image.size == (100, 50) and mask.shape == (50, 100)
    assert set(np.unique(mask)) == {0, 255} and mask[:, :50].all() and not mask[:, 50:].any()


def test_isic_preparation_rejects_misaligned_mask(tmp_path):
    from download_datasets import prepare_isic
    Image.new('RGB', (40, 20)).save(tmp_path / 'ISIC_0000001.jpg')
    Image.new('L', (20, 20)).save(tmp_path / 'ISIC_0000001_segmentation.png')
    with pytest.raises(ValueError, match='size mismatch'):
        prepare_isic(tmp_path, tmp_path, tmp_path / 'out', 10)


def test_task3_overlap_reports_shared_ids(tmp_path):
    from download_datasets import task3_overlap
    path = tmp_path / 'task3.csv'
    path.write_text('image,MEL\nISIC_1,1\nISIC_2,0\n')
    assert task3_overlap(['ISIC_2', 'ISIC_3'], path) == ['ISIC_2']


def test_paco_fetch_rejects_non_jpeg_response(tmp_path):
    from download_datasets import fetch_image
    source = tmp_path / 'page.html'
    source.write_text('<html>error</html>')
    record = {'file_name': 'train2017/a.jpg', 'coco_url': source.as_uri()}
    with pytest.raises(ValueError, match='Not a JPEG'):
        fetch_image(record, tmp_path / 'images')
    assert not (tmp_path / 'images/train2017/a.jpg').exists()


def tls_failure(monkeypatch, content):
    import io
    import ssl
    import urllib.error
    import download_datasets as module

    def reject(*args, **kwargs):
        raise urllib.error.URLError(ssl.SSLCertVerificationError('incomplete chain'))
    monkeypatch.setattr(module, 'download', reject)
    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(content))
    return module


def test_pinned_download_survives_incomplete_tls_chain(tmp_path, monkeypatch):
    import hashlib
    module = tls_failure(monkeypatch, b'archive')
    digest = hashlib.sha256(b'archive').hexdigest()
    target = tmp_path / 'a.zip'
    assert module.download_pinned('https://example.test/a.zip', target, digest) == target
    assert target.read_bytes() == b'archive'


def test_pinned_download_never_keeps_unverified_bytes(tmp_path, monkeypatch):
    module = tls_failure(monkeypatch, b'tampered')
    target = tmp_path / 'a.zip'
    with pytest.raises(ValueError, match='checksum'):
        module.download_pinned('https://example.test/a.zip', target, '0' * 64)
    assert list(tmp_path.iterdir()) == []


def test_model_downloads_are_fully_pinned():
    from download_datasets import DATASETS, MODELS
    assert set(MODELS) == set(DATASETS)
    for name, files in MODELS.items():
        for url, _, digest in files:
            assert url.startswith('https://') and len(digest) >= 8
            if 'huggingface.co' in url:
                assert '/resolve/main/' not in url and len(digest) == 64


@pytest.mark.parametrize('runner,dataset', RUNNERS.items())
def test_runners_are_registered(runner, dataset):
    import budget_ablation
    import experiment_common as common
    import prepare_full_budget_study as study
    from summarize_ablation_study import DATASETS
    config = yaml.safe_load((ROOT / f'src/configs/{dataset}_epito.yaml').read_text())
    assert config['data']['name'].replace('_', '').lower() == common.EXPECTED_DATASETS[runner]
    assert config['data']['mask_dir_final'].startswith(f'{dataset}_sam2_')
    assert common.DEFAULT_CONFIGS[runner] == f'{dataset}_epito' and runner in budget_ablation.RUNNERS
    assert runner in DATASETS and study.MODELS[dataset][0][0] == runner
    assert (ROOT / f'src/scripts/explanations/{runner}.py').is_file()
    assert (ROOT / f'src/scripts/generate_{dataset.lower()}_partitions.py').is_file()
    assert common.create_parser(runner).parse_args([]).config == f'{dataset}_epito'


def test_full_budget_default_needs_no_new_partitions():
    import prepare_full_budget_study as study
    assert not set(study.DEFAULT_DATASETS) & set(RUNNERS.values())
    assert set(study.DEFAULT_DATASETS) <= set(study.MODELS)


def test_checkpoint_runner_without_checkpoint_names_the_download(tmp_path):
    import experiment_common as common
    images = tmp_path / 'images'
    folder = tmp_path / 'partitions/KvasirSEG_test'
    images.mkdir()
    folder.mkdir(parents=True)
    Image.new('RGB', (8, 8)).save(images / 'abc.jpg')
    np.save(folder / 'abc_compact.npy', np.zeros((8, 8), dtype=np.int32))
    (folder / 'abc_results.json').write_text(json.dumps({'saved_partitions': {'compact': 'x'}}))
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({
        'data': {'name': 'KvasirSEG', 'input_format': 'directory', 'image_dir': str(images),
                 'mask_dir_final': folder.name},
        'output': {'partitions': str(folder.parent), 'explanations': str(tmp_path / 'e'),
                   'evaluations': str(tmp_path / 'v')}}))
    args = common.create_parser('OC_KvasirSEG').parse_args(['--config', str(path)])
    with pytest.raises(ValueError, match='download_datasets.py --dataset kvasirseg'):
        common.prepare_inputs('OC_KvasirSEG', args)
