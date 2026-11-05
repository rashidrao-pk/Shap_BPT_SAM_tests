"""BUSI discovery and lesion alignment regression checks."""
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.medical.datasets import discover_images, lesion_mask
from src.medical.grayscale_adapter import to_rgb
from src.medical.evaluation import pointing_accuracy


def test_masks_excluded_united_without_flip(tmp_path):
    folder = tmp_path / 'benign'
    folder.mkdir()
    image = np.arange(24, dtype=np.uint8).reshape(4, 6)
    Image.fromarray(image).save(folder / 'benign (1).png')
    for suffix, position in [('_mask', (0, 1)), ('_mask_1', (3, 4))]:
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[position] = 255
        Image.fromarray(mask).save(folder / f'benign (1){suffix}.png')
    records = discover_images(tmp_path)
    assert len(records) == 1
    union, paths = lesion_mask(tmp_path / records[0]['file_name'])
    assert len(paths) == 2 and union.sum() == 2
    assert union[0, 1] and union[3, 4]
    rgb = np.asarray(to_rgb(image))
    assert np.array_equal(rgb[..., 0], image)
    assert np.array_equal(rgb[..., 0], rgb[..., 2])
    assert pointing_accuracy(union.astype(float), union) == 1
    assert pointing_accuracy(image, np.zeros_like(union)) is None


def test_frozen_classifier_and_probability_scores(monkeypatch):
    import types
    import torch
    from src.medical.classifier import BusiAdapter

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def __call__(self, images, return_tensors):
            return {'pixel_values': torch.tensor(np.asarray(images).copy()).permute(2, 0, 1).float()[None] / 255}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = types.SimpleNamespace(num_labels=3, id2label={0: 'benign', 1: 'malignant', 2: 'normal'})
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
        def forward(self, pixel_values):
            mean = pixel_values.mean(dim=(1, 2, 3)) * self.weight
            return types.SimpleNamespace(logits=torch.stack([mean, -mean, mean * 0], dim=1))

    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(
        AutoImageProcessor=Processor, AutoModelForImageClassification=Model))
    adapter = BusiAdapter(types.SimpleNamespace(model='stub', checkpoint=None,
                          local_files_only=True, model_batch_size=1), 'cpu')
    result = adapter.predict_batch([np.full((4, 6), 100, dtype=np.uint8)] * 2)
    assert result.shape == (2, 3)
    assert np.allclose(result.sum(axis=1), 1)
    assert not adapter.model.training
    assert all(not p.requires_grad for p in adapter.model.parameters())
    tensor = adapter.preprocess(np.full((4, 6), 100, dtype=np.uint8))[None].requires_grad_(True)
    adapter.forward_tensor(tensor)[0, 0].backward()
    assert tensor.grad is not None  # Frozen weights still allow input attribution.


def test_shared_busi_discovery_and_ground_truth(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
    from generate_coco_partitions import collect_images
    from ground_truth import load_ground_truth
    folder = tmp_path / 'malignant'
    folder.mkdir()
    Image.fromarray(np.zeros((4, 6), dtype=np.uint8)).save(folder / 'malignant (1).png')
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[0, 2] = 255
    Image.fromarray(mask).save(folder / 'malignant (1)_mask.png')
    records = collect_images({'name': 'BUSI', 'image_dir': str(tmp_path), 'input_format': 'directory'})
    assert len(records) == 1
    config = {'data': {'name': 'BUSI', 'ground_truth': {'format': 'busi', 'root': str(tmp_path)}}}
    gt = load_ground_truth(config, folder / 'malignant (1).png', (4, 6, 3), predicted_class='benign')
    assert gt['mask'][0, 2] and gt['foreground_pixels'] == 1
    assert gt['scope'] == 'lesion_region' and gt['dataset_class'] == 'malignant'


@pytest.mark.parametrize('all_methods', [False, True])
def test_busi_explanation_pipeline_with_stub_classifier(tmp_path, monkeypatch, all_methods):
    import json
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'))
    import experiment_common as common
    image_dir = tmp_path / 'images' / 'benign'
    image_dir.mkdir(parents=True)
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    Image.fromarray(image).save(image_dir / 'benign (1).png')
    Image.fromarray((image > 128).astype(np.uint8) * 255).save(image_dir / 'benign (1)_mask.png')
    partitions = tmp_path / 'partitions' / 'BUSI_test'
    partitions.mkdir(parents=True)
    labels = (image > 128).astype(np.uint16)
    np.save(partitions / 'benign (1)_compact.npy', labels)
    (partitions / 'benign (1)_results.json').write_text(json.dumps({'saved_partitions': {'compact': 'benign (1)_compact.npy'}}))
    config = {'device': 'cpu', 'data': {'name': 'BUSI', 'input_format': 'directory',
              'image_dir': str(image_dir.parent), 'mask_dir_final': 'BUSI_test',
              'ground_truth': {'format': 'busi', 'root': str(image_dir.parent)}},
              'output': {key: str(tmp_path / key) for key in ['partitions', 'explanations', 'evaluations']}}
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.safe_dump(config))

    class Adapter:
        class_names = ['benign', 'malignant', 'normal']
        def predict_batch(self, images):
            means = np.asarray([np.asarray(x).mean() / 255 for x in images])
            return np.column_stack([means, (1-means)*.75, (1-means)*.25])
        def detections(self, image):
            return []
    monkeypatch.setattr(common, 'load_adapter', lambda *args: Adapter())
    arguments = ['--config', str(config_file), '--show-gt', '--budget', '8',
        '--no-plots', '--no-html', '--torch-threads', '1']
    arguments += (['--explain_all', '--lime-budgets', '8', '--aa-budget', '8', '--auc-num-samples', '3']
                  if all_methods else ['--no-auc'])
    args = common.create_parser('OC_BUSI').parse_args(arguments)
    common.run('OC_BUSI', args)
    metrics = list((tmp_path / 'explanations').rglob('*_metrics.json'))
    assert len(metrics) == 1
    rows = json.loads(metrics[0].read_text())
    assert len(rows) == (5 if all_methods else 3)
    assert all('pointing_accuracy' in row for row in rows)
    if all_methods:
        assert all(row['auc_requested_points'] == 3 for row in rows)


def test_model_card_preprocessing_without_processor_metadata(monkeypatch):
    import types
    import torch
    from src.medical.classifier import BusiAdapter
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(num_labels=3, id2label={0: 'benign', 1: 'malignant', 2: 'normal'})
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoModelForImageClassification=Model))
    adapter = BusiAdapter(types.SimpleNamespace(checkpoint=Path('local_model_without_processor'),
        model='unused', local_files_only=True, model_batch_size=1), 'cpu')
    value = adapter.preprocess(np.full((20, 30), 255, dtype=np.uint8))
    assert tuple(value.shape) == (3, 224, 224)
    assert torch.allclose(value, torch.ones_like(value))
    assert torch.allclose(adapter.preprocess(np.zeros((20, 30), dtype=np.uint8)), -torch.ones_like(value))


def test_siglip_gradcam_patch_alignment():
    import torch
    from types import SimpleNamespace
    from torchvision.transforms import Compose, Resize, ToTensor
    from transformers import SiglipConfig, SiglipForImageClassification
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'))
    from all_methods import GradientMethods, UnsupportedMethod
    from experiment_common import create_parser
    import pytest
    torch.set_num_threads(1)
    config = SiglipConfig(vision_config={'hidden_size': 8, 'intermediate_size': 16,
        'num_hidden_layers': 1, 'num_attention_heads': 2, 'image_size': 16, 'patch_size': 8}, num_labels=3)
    config._attn_implementation = 'eager'
    model = SiglipForImageClassification(config).eval().requires_grad_(False)
    adapter = SimpleNamespace(name='siglip_busi', model=model, device='cpu',
        preprocessing_geometry='resize', preprocess=Compose([Resize((16, 16)), ToTensor()]),
        forward_tensor=lambda x: model(pixel_values=x).logits.softmax(dim=1))
    image = np.arange(20 * 24 * 3, dtype=np.uint8).reshape(20, 24, 3)
    arguments = create_parser('OC_BUSI').parse_args(['--gradient-samples', '2', '--idg-steps', '4',
                                                       '--attribution-batch-size', '2'])
    methods = GradientMethods(adapter, image, [np.zeros_like(image)], arguments)
    for method in ['GradCAM', 'IDG', 'GradExp', 'GradShap']:
        values = methods.compute(method, 0)
        assert values.shape == (1, 20, 24) and np.isfinite(values).all()
    assert not model.vision_model.encoder.layers[-1].layer_norm1._forward_hooks
    with pytest.raises(UnsupportedMethod):
        methods.compute('LRP', 0)
