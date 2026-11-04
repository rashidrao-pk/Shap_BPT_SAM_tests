"""Frozen SigLIP BUSI classifier, using checkpoint preprocessing and label order."""
import numpy as np
from .grayscale_adapter import to_rgb

MODEL_ID = 'ghostsas001/busi-breast-ultrasound-siglip'


class BusiAdapter:
    def __init__(self, args, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.device, self.batch_size = device, args.model_batch_size
        source = str(args.checkpoint) if args.checkpoint else args.model
        options = {'local_files_only': args.local_files_only}
        self.processor = AutoImageProcessor.from_pretrained(source, **options)
        self.model = AutoModelForImageClassification.from_pretrained(source, **options).to(device).eval()
        self.model.requires_grad_(False)
        labels = self.model.config.id2label
        self.class_names = [labels.get(i, labels.get(str(i))) for i in range(self.model.config.num_labels)]
        if set(self.class_names) != {'benign', 'malignant', 'normal'}:
            raise ValueError(f'Expected BUSI three-class checkpoint, found {self.class_names}')
        self.name, self.score_type = 'siglip_busi', 'classification probability'
        self.preprocessing_geometry = 'resize'

    def preprocess(self, image):
        return self.processor(images=to_rgb(image), return_tensors='pt')['pixel_values'][0]

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
