"""Frozen classifiers whose training preprocessing differs from torchvision ImageNet presets."""
import json
from pathlib import Path

import numpy as np


class ResizeClassifier:
    """Softmax classifier using a direct resize, so attribution maps invert by resizing."""
    preprocessing_geometry = 'resize'
    score_type = 'classification probability'

    def _finish(self, args, device, model, transform, class_names, name):
        self.device, self.batch_size = device, args.model_batch_size
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.transform, self.class_names, self.name = transform, class_names, name

    def preprocess(self, image):
        return self.transform(image.convert('RGB'))

    def logits(self, tensor):
        return self.model(tensor)

    def forward_tensor(self, tensor):
        return self.logits(tensor).softmax(dim=1)

    def predict_batch(self, images):
        import torch
        from PIL import Image
        rows = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.stack([self.preprocess(Image.fromarray(np.asarray(image, dtype=np.uint8)))
                                     for image in images[start:start + self.batch_size]])
                rows.append(self.forward_tensor(batch.to(self.device)).cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0, len(self.class_names)))

    def detections(self, image):
        return []


class FunnyBirdsAdapter(ResizeClassifier):
    """Authors' ResNet-50: 256x256 ToTensor input, without ImageNet normalization."""

    def __init__(self, args, device):
        import torch
        from torchvision.models import resnet50
        from torchvision.transforms import Compose, Resize, ToTensor
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        state = {key.removeprefix('module.').removeprefix('model.'): value
                 for key, value in checkpoint['state_dict'].items()}
        classes = state['fc.weight'].shape[0]
        if classes != 50:
            raise ValueError(f'Expected the 50-class FunnyBirds ResNet-50, found {classes} classes')
        model = resnet50(weights=None, num_classes=classes)
        model.load_state_dict(state, strict=True)
        # The name selects the existing ResNet-50 GradCAM and LRP paths.
        self._finish(args, device, model, Compose([Resize((256, 256)), ToTensor()]),
                     [f'class_{index}' for index in range(classes)], 'resnet50')


class KvasirAdapter(ResizeClassifier):
    """ViT fine-tuned on Kvasir v2, with the processor settings saved beside the checkpoint."""

    def __init__(self, args, device):
        from transformers import AutoModelForImageClassification
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        source = str(args.checkpoint) if args.checkpoint else args.model
        model = AutoModelForImageClassification.from_pretrained(source, local_files_only=args.local_files_only)
        labels = model.config.id2label
        names = [labels.get(index, labels.get(str(index))) for index in range(model.config.num_labels)]
        if 'polyps' not in names or model.config.image_size != 224:
            raise ValueError(f'Expected the 224-pixel Kvasir v2 checkpoint, found {names}')
        transform = Compose([Resize((224, 224)), ToTensor(), Normalize([.5] * 3, [.5] * 3)])
        self._finish(args, device, model, transform, names, 'vit_kvasir')

    def logits(self, tensor):
        return self.model(pixel_values=tensor).logits


def efficientnet_b0(classes, dropout):
    """Module layout of the published ISIC weights: encoder, dropout, classifier."""
    from torch import nn
    from torchvision import models

    class EfficientNetB0(nn.Module):
        def __init__(self):
            super().__init__()
            network = models.efficientnet_b0(weights=None)
            self.encoder = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(network.classifier[1].in_features, classes)

        def forward(self, image):
            return self.classifier(self.dropout(self.encoder(image)))

    return EfficientNetB0()


class IsicAdapter(ResizeClassifier):
    """ISIC 2018 Task 3 EfficientNet-B0; class order and transforms come from its config.json."""

    def __init__(self, args, device):
        from safetensors.torch import load_file
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        folder = Path(args.checkpoint)
        config = json.loads((folder / 'config.json').read_text())
        names, size = config['class_names'], int(config['image_size'])
        if set(names) != {'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'}:
            raise ValueError(f'Expected the seven ISIC 2018 Task 3 classes, found {names}')
        model = efficientnet_b0(len(names), float(config['dropout']))
        model.load_state_dict(load_file(folder / 'model.safetensors'), strict=True)
        norm = config['normalization']
        transform = Compose([Resize((size, size)), ToTensor(), Normalize(norm['mean'], norm['std'])])
        self._finish(args, device, model, transform, names, 'efficientnet_b0_isic')


ADAPTERS = {'OC_FunnyBirds': FunnyBirdsAdapter, 'OC_KvasirSEG': KvasirAdapter, 'OC_ISIC2018': IsicAdapter}
