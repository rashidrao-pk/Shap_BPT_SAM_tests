"""Model-specific scoring for the shared explanation runner (imports are lazy)."""
import json
from pathlib import Path


def read_class_names(value):
    if value is None:
        return None
    names = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    if isinstance(names, dict):
        names = [names[str(index)] if str(index) in names else names[index]
                 for index in range(len(names))]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError('Class names must be a nonempty JSON list or zero-indexed dictionary')
    return names


class ClassifierAdapter:
    def __init__(self, args, device, architecture):
        import torch
        from torchvision.models import (ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16)
        self.torch, self.device, self.batch_size = torch, device, args.model_batch_size
        enum = ResNet50_Weights if architecture == 'resnet50' else ViT_B_16_Weights
        builder = resnet50 if architecture == 'resnet50' else vit_b_16
        weights = None if args.weights.lower() == 'none' else getattr(enum, args.weights)
        default_weights = weights or enum.DEFAULT
        names = read_class_names(args.class_names)
        state, metadata = None, {}
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            metadata = checkpoint if isinstance(checkpoint, dict) else {}
            state = metadata.get('state_dict', metadata.get('model', checkpoint))
            if not isinstance(state, dict):
                raise ValueError('Checkpoint must contain a torchvision-compatible state_dict')
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            names = names or read_class_names(metadata.get('class_names', metadata.get('categories')))
        head = 'fc.weight' if architecture == 'resnet50' else 'heads.head.weight'
        classes = int(state[head].shape[0]) if state is not None and head in state else 1000
        if names is not None and len(names) != classes:
            raise ValueError(f'{len(names)} class names supplied for a {classes}-class checkpoint')
        self.class_names = names or (list(default_weights.meta['categories']) if classes == 1000
                                    else [f'class_{index}' for index in range(classes)])
        # A complete checkpoint supplies the backbone too; do not download a second copy.
        self.model = builder(weights=None if state is not None else weights, num_classes=classes)
        if state is not None:
            self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.preprocess = default_weights.transforms()
        self.name = architecture

    def predict_batch(self, images):
        import numpy as np
        from PIL import Image
        torch = self.torch
        if not len(images):
            return np.empty((0, len(self.class_names)))
        results = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.stack([self.preprocess(Image.fromarray(np.asarray(image, dtype=np.uint8)))
                                     for image in images[start:start + self.batch_size]]).to(self.device)
                results.append(torch.softmax(self.model(batch), dim=1).cpu().numpy())
        return np.concatenate(results)

    def detections(self, image):
        return []


def detection_scores(result, class_count, aggregate):
    import numpy as np
    scores = np.zeros(class_count, dtype=float)
    for label, score in zip(result['labels'].detach().cpu().numpy(), result['scores'].detach().cpu().numpy()):
        label = int(label)
        if 0 <= label < class_count:
            scores[label] = max(scores[label], float(score)) if aggregate == 'max' else scores[label] + float(score)
    return scores


class DetrAdapter:
    def __init__(self, args, device):
        import torch
        from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
        self.torch, self.device, self.batch_size = torch, device, args.model_batch_size
        self.threshold, self.aggregate = args.threshold, args.class_aggregation
        source = str(args.checkpoint) if args.checkpoint else args.model
        options = {'local_files_only': args.local_files_only}
        config = DetrConfig.from_pretrained(source, **options)
        config.use_pretrained_backbone = False  # Included in the detector checkpoint already.
        self.processor = DetrImageProcessor.from_pretrained(source, **options)
        if args.detr_shortest_edge:
            self.processor.size = {'shortest_edge': args.detr_shortest_edge, 'longest_edge': 1333}
        self.model = DetrForObjectDetection.from_pretrained(source, config=config, **options).to(device).eval()
        labels = {int(key): str(value) for key, value in self.model.config.id2label.items()}
        self.class_names = [labels.get(index, f'class_{index}') for index in range(max(labels) + 1)]
        self.padding_classes = [index for index, name in enumerate(self.class_names) if name.upper() == 'N/A']
        self.name = source

    def _results(self, images):
        from PIL import Image
        import numpy as np
        pil = [Image.fromarray(np.asarray(image, dtype=np.uint8)) for image in images]
        inputs = self.processor(images=pil, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            sizes = self.torch.tensor([image.size[::-1] for image in pil], device=self.device)
            return self.processor.post_process_object_detection(outputs, threshold=self.threshold,
                                                                 target_sizes=sizes)

    def predict_batch(self, images):
        import numpy as np
        rows = []
        for start in range(0, len(images), self.batch_size):
            rows.extend(detection_scores(result, len(self.class_names), self.aggregate)
                        for result in self._results(images[start:start + self.batch_size]))
        scores = np.stack(rows) if rows else np.empty((0, len(self.class_names)))
        scores[:, self.padding_classes] = 0
        return scores

    def detections(self, image):
        result = self._results([image])[0]
        return [{'class_id': int(label), 'class_name': self.class_names[int(label)],
                 'confidence': float(score), 'box': box.tolist()}
                for label, score, box in zip(result['labels'].cpu(), result['scores'].cpu(), result['boxes'].cpu())
                if float(score) >= max(.3, self.threshold) and int(label) not in self.padding_classes]


def load_adapter(kind, args, device):
    if kind == 'OD_DETR':
        return DetrAdapter(args, device)
    return ClassifierAdapter(args, device, 'resnet50' if kind == 'OC_ResNet' else 'vit_b_16')
