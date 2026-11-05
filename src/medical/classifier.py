"""Frozen SigLIP BUSI classifier, using the BUSI model-card preprocessing and checkpoint label order."""
import numpy as np
from .grayscale_adapter import to_rgb

MODEL_ID = 'ghostsas001/busi-breast-ultrasound-siglip'


class BusiAdapter:
    def __init__(self, args, device):
        import torch
        from transformers import AutoModelForImageClassification
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize
        self.device, self.batch_size = device, args.model_batch_size
        source = str(args.checkpoint) if args.checkpoint else args.model
        options = {'local_files_only': args.local_files_only}
        self.model = AutoModelForImageClassification.from_pretrained(source, **options).to(device).eval()
        self.model.requires_grad_(False)
        labels = self.model.config.id2label
        self.class_names = [labels.get(i, labels.get(str(i))) for i in range(self.model.config.num_labels)]
        if set(self.class_names) != {'benign', 'malignant', 'normal'}:
            raise ValueError(f'Expected BUSI three-class checkpoint, found {self.class_names}')
        size = getattr(getattr(self.model.config, 'vision_config', None), 'image_size', 224)
        if size != 224:
            raise ValueError('BUSI SigLIP model-card preprocessing requires a 224-pixel checkpoint')
        self.transform = Compose([Resize((224, 224)), ToTensor(), Normalize([.5] * 3, [.5] * 3)])
        self.name, self.score_type = 'siglip_busi', 'classification probability'
        self.preprocessing_geometry = 'resize'

    def preprocess(self, image):
        return self.transform(to_rgb(image))

    def forward_tensor(self, tensor):
        return self.model(pixel_values=tensor).logits.softmax(dim=1)

    def predict_batch(self, images):
        import torch
        rows = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.stack([self.preprocess(image) for image in images[start:start+self.batch_size]])
                rows.append(self.forward_tensor(batch.to(self.device)).cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0, len(self.class_names)))

    def detections(self, image):
        return []
