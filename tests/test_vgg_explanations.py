"""VGG checkpoint routing and input-gradient methods without downloading weights."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'))
from experiment_common import create_parser
from model_adapters import ClassifierAdapter
from all_methods import GradientMethods


def test_vgg_custom_checkpoint_head_and_labels():
    args = create_parser('OC_VGG').parse_args(['--checkpoint', '/fixture/model.pt'])
    model = Mock()
    state = {'classifier.6.weight': torch.zeros(2, 4096)}
    with patch('torch.load', return_value={'state_dict': state, 'class_names': ['one', 'two']}):
        with patch('torchvision.models.vgg16', return_value=model) as builder:
            adapter = ClassifierAdapter(args, 'cpu', 'vgg16')
    builder.assert_called_once_with(weights=None, num_classes=2)
    model.load_state_dict.assert_called_once_with(state, strict=True)
    assert adapter.name == 'vgg16' and adapter.class_names == ['one', 'two']


def test_feature_only_weights_rejected():
    args = create_parser('OC_VGG').parse_args(['--weights', 'IMAGENET1K_FEATURES'])
    with pytest.raises(ValueError, match='feature-only'):
        ClassifierAdapter(args, 'cpu', 'vgg16')


def test_vgg_gradient_methods_on_small_vgg_architecture():
    from torchvision.transforms import Compose, Resize, ToTensor
    torch.set_num_threads(1)
    class SmallVGG(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(2))
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = torch.nn.Sequential(torch.nn.Linear(4, 2))
        def forward(self, x):
            return self.classifier(self.pool(self.features(x)).flatten(1))
    model = SmallVGG().eval().requires_grad_(False)
    adapter = SimpleNamespace(name='vgg16', device='cpu', model=model,
        preprocess=Compose([Resize((16, 16)), ToTensor()]), preprocessing_geometry='resize')
    args = create_parser('OC_VGG').parse_args(['--explain_all', '--gradient-samples', '2'])
    image = np.full((20, 24, 3), 100, dtype=np.uint8)
    methods = GradientMethods(adapter, image, [np.zeros_like(image)], args)
    for name in ['GradCAM', 'LRP']:
        values = methods.compute(name, 0)
        assert values.shape == (1, 20, 24) and np.isfinite(values).all()
