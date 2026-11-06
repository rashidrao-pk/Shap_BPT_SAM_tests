"""GradCAM must hook pre-attention patch tokens in old and new HF ViT layouts."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torchvision.transforms import Compose, Resize, ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/scripts/explanations'))
from all_methods import GradientMethods, UnsupportedMethod, vit_gradcam_layer
from experiment_common import create_parser


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm_before = torch.nn.LayerNorm(8)

    def forward(self, tokens):
        features = self.layernorm_before(tokens)
        # CLS must depend on patch tokens, as in attention.
        return tokens + features.mean(dim=1, keepdim=True)


class Classifier(torch.nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.config = SimpleNamespace(patch_size=8)
        self.embedding = torch.nn.Conv2d(3, 8, 8, stride=8)
        self.vit = torch.nn.Module()
        blocks = torch.nn.ModuleList([Block()])
        if layout == 'legacy':
            self.vit.encoder = torch.nn.Module()
            self.vit.encoder.layer = blocks
        else:
            self.vit.layers = blocks
        self.head = torch.nn.Linear(8, 2)

    def forward(self, pixel_values):
        tokens = self.embedding(pixel_values).flatten(2).transpose(1, 2)
        tokens = torch.cat([torch.zeros_like(tokens[:, :1]), tokens], dim=1)
        blocks = self.vit.encoder.layer if hasattr(self.vit, 'encoder') else self.vit.layers
        for block in blocks:
            tokens = block(tokens)
        return SimpleNamespace(logits=self.head(tokens[:, 0]))


@pytest.mark.parametrize('layout', ['legacy', 'flat'])
def test_kvasir_gradcam_and_other_gradients_preserve_geometry_and_remove_hooks(layout):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = Classifier(layout).eval().requires_grad_(False)
    adapter = SimpleNamespace(model=model, name='vit_kvasir', device=torch.device('cpu'),
        preprocessing_geometry='resize', preprocess=Compose([Resize((16, 24)), ToTensor()]),
        forward_tensor=lambda x: model(pixel_values=x).logits.softmax(dim=1))
    args = create_parser('OC_KvasirSEG').parse_args(['--gradient-samples', '2', '--idg-steps', '4',
                                                  '--attribution-batch-size', '2'])
    image = np.arange(20*30*3, dtype=np.uint8).reshape(20, 30, 3)
    engine = GradientMethods(adapter, image, [np.zeros_like(image)], args)
    layer = vit_gradcam_layer(model, adapter.name)
    for method in ['GradCAM', 'IDG', 'GradExp', 'GradShap']:
        values = engine.compute(method, 1)
        assert values.shape == (1, 20, 30)
        assert np.isfinite(values).all()
        assert not layer._forward_hooks


def test_unknown_vit_layout_is_truthfully_unsupported():
    with pytest.raises(UnsupportedMethod, match='pre-attention layernorm'):
        vit_gradcam_layer(SimpleNamespace(vit=SimpleNamespace()), 'vit_kvasir')


def test_torchvision_layer_selection_is_unchanged():
    norm = torch.nn.LayerNorm(8)
    model = SimpleNamespace(encoder=SimpleNamespace(layers=[SimpleNamespace(ln_1=norm)]))
    assert vit_gradcam_layer(model, 'vit_b_16') is norm
