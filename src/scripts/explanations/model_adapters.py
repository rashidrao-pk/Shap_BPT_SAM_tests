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
        from torchvision.models import (ResNet50_Weights, ViT_B_16_Weights, VGG16_Weights,
                                        resnet50, vit_b_16, vgg16)
        self.torch, self.device, self.batch_size = torch, device, args.model_batch_size
        architectures = {'resnet50': (ResNet50_Weights, resnet50, 'fc.weight'),
                         'vit_b_16': (ViT_B_16_Weights, vit_b_16, 'heads.head.weight'),
                         'vgg16': (VGG16_Weights, vgg16, 'classifier.6.weight')}
        if architecture not in architectures:
            raise ValueError(f'Unsupported classifier architecture: {architecture}')
        enum, builder, head = architectures[architecture]
        weights = None if args.weights.lower() == 'none' else getattr(enum, args.weights)
        default_weights = weights or enum.DEFAULT
        if default_weights.meta.get('categories') is None:
            raise ValueError('Choose classification weights; feature-only weights have no valid classifier')
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
        classes = int(state[head].shape[0]) if state is not None and head in state else 1000
        if names is not None and len(names) != classes:
            raise ValueError(f'{len(names)} class names supplied for a {classes}-class checkpoint')
        self.class_names = names or (list(default_weights.meta['categories']) if classes == 1000
                                    else [f'class_{index}' for index in range(classes)])
        self.label_space = metadata.get('label_space') or ('imagenet1k' if state is None else None)
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


YOLO_FOLDERS = ('src/notebooks/checkpoints', 'src/checkpoints', 'checkpoints')  # Same search as OD_yolo.py


def resolve_yolo_weights(checkpoint, model, root=None):
    """--checkpoint when given, else <model>.pt in the repository's checkpoint folders."""
    if checkpoint:
        return Path(checkpoint)
    root = Path(root or Path(__file__).resolve().parents[3])
    name = model if str(model).endswith('.pt') else f'{model}.pt'
    for folder in YOLO_FOLDERS:
        if (root / folder / name).is_file():
            return root / folder / name
    raise FileNotFoundError(f'{name} not found in {", ".join(YOLO_FOLDERS)}; '
                            'run download_datasets.py --dataset dota --assets model or pass --checkpoint')


def _numpy(value):
    import numpy as np
    value = value.detach().cpu().numpy() if hasattr(value, 'detach') else value
    return np.asarray(value)


def yolo_device(device):
    """'cuda:N', 'cpu' or 'mps'. Ultralytics 8.3 strips only 'cuda:' and would misread a bare 'cuda'."""
    device = str(device)
    if device.startswith('cuda'):
        return device if ':' in device else 'cuda:0'
    return device


def obb_scores(result, class_count, aggregate):
    """Per class, the sum (or max) of the oriented-box confidences of one Ultralytics result.

    OBB results keep detections in result.obb; result.boxes is None for them.
    """
    import numpy as np
    scores = np.zeros(class_count, dtype=float)
    obb = getattr(result, 'obb', None)
    if obb is None or len(obb) == 0:
        return scores
    for label, score in zip(_numpy(obb.cls).astype(int).ravel(), _numpy(obb.conf).astype(float).ravel()):
        if 0 <= label < class_count:
            scores[label] = max(scores[label], score) if aggregate == 'max' else scores[label] + score
    return scores


class YoloObbAdapter:
    """Ultralytics YOLO11-OBB: a class's score sums the confidences of its oriented boxes on the image."""
    image_size = 1024  # The DOTA training tile size; tiles are prepared at this size.

    def __init__(self, args, device, model=None):
        self.device, self.batch_size = device, args.model_batch_size
        self.conf, self.aggregate = args.conf, args.class_aggregation
        if model is None:
            from ultralytics import YOLO
            weights = resolve_yolo_weights(args.checkpoint, args.model)
            model = YOLO(str(weights))
            self.name = str(weights)
        else:
            self.name = getattr(model, 'ckpt_path', None) or args.model
        if getattr(model, 'task', None) != 'obb':
            raise ValueError(f'{self.name}: task {getattr(model, "task", None)!r}; OD_DOTA needs an oriented-box (obb) model')
        names = model.names
        self.class_names = [str(names[index]) for index in range(len(names))] if isinstance(names, dict) else list(names)
        self.model = model

    def _results(self, images):
        import numpy as np
        # Ultralytics reads NumPy arrays as BGR (OpenCV order); the runner passes RGB.
        bgr = [np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[..., ::-1]) for image in images]
        return self.model.predict(source=bgr, imgsz=self.image_size, conf=self.conf, half=False, augment=False,
                                  batch=len(bgr), device=yolo_device(self.device), verbose=False)

    def predict_batch(self, images):
        import numpy as np
        rows = []
        for start in range(0, len(images), self.batch_size):
            rows.extend(obb_scores(result, len(self.class_names), self.aggregate)
                        for result in self._results(images[start:start + self.batch_size]))
        return np.stack(rows) if rows else np.empty((0, len(self.class_names)))

    def detections(self, image):
        result = self._results([image])[0]
        obb = getattr(result, 'obb', None)
        if obb is None or len(obb) == 0:
            return []
        found = []
        for label, score, corners in zip(_numpy(obb.cls).astype(int).ravel(), _numpy(obb.conf).astype(float).ravel(),
                                         _numpy(obb.xyxyxyxy).reshape(-1, 4, 2)):
            if score >= max(.3, self.conf) and 0 <= label < len(self.class_names):
                found.append({'class_id': int(label), 'class_name': self.class_names[label], 'confidence': float(score),
                              'polygon': corners.astype(float).tolist(),
                              'box': [float(corners[:, 0].min()), float(corners[:, 1].min()),
                                      float(corners[:, 0].max()), float(corners[:, 1].max())]})
        return found


def clip_size(crop):
    """(height, width) from crop_size: a dict, a transformers SizeDict (newer versions), or an int."""
    if isinstance(crop, int):
        return crop, crop
    if isinstance(crop, dict):
        return int(crop['height']), int(crop['width'])
    height, width = getattr(crop, 'height', None), getattr(crop, 'width', None)
    if height is None or width is None:
        side = getattr(crop, 'shortest_edge', None)
        if side is None:
            raise ValueError(f'Unsupported CLIP crop_size: {crop!r}')
        return int(side), int(side)
    return int(height), int(width)


class ClipAdapter:
    """CLIP image-text cosine similarity for the current referring expression: one output.

    The runner calls set_record() before each record. Images are resized whole to the model's
    224x224 input instead of CLIP's usual center crop, so objects near the border stay visible.
    """
    preprocessing_geometry = 'resize'  # Gradient maps are resized straight back to the image.
    name = 'clip_vit'

    def __init__(self, args, device, model=None, processor=None):
        import torch
        from torchvision import transforms
        self.torch, self.device, self.batch_size = torch, device, args.model_batch_size
        source = str(args.checkpoint) if args.checkpoint else args.model
        if model is None or processor is None:
            from transformers import CLIPModel, CLIPProcessor
            options = {'local_files_only': bool(args.checkpoint) or args.local_files_only}
            processor = CLIPProcessor.from_pretrained(source, **options)
            model = CLIPModel.from_pretrained(source, **options)
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.tokenizer = processor.tokenizer
        settings = processor.image_processor
        crop = getattr(settings, 'crop_size', None) or {'height': 224, 'width': 224}
        size = clip_size(crop)
        self.preprocess = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(), transforms.Normalize(settings.image_mean, settings.image_std)])
        self.source = source
        self.text_features, self.expression = None, None
        self.class_names = ['<no expression set>']

    def set_record(self, record):
        expression = record['expression']
        if expression == self.expression:
            return
        tokens = self.tokenizer([expression], padding=True, truncation=True, return_tensors='pt')
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with self.torch.inference_mode():
            features = self._features(self.model.get_text_features(**tokens))
        self.text_features = features / features.norm(dim=-1, keepdim=True)
        self.expression, self.class_names = expression, [expression]

    def _features(self, output):
        # Transformers 4 returns the projected embedding; newer versions wrap it in an output object.
        return output if isinstance(output, self.torch.Tensor) else output.pooler_output

    def forward_tensor(self, tensor):
        if self.text_features is None:
            raise RuntimeError('ClipAdapter.set_record() must be called before scoring')
        image = self._features(self.model.get_image_features(pixel_values=tensor))
        image = image / image.norm(dim=-1, keepdim=True)
        return image @ self.text_features.T.to(image.dtype)

    def predict_batch(self, images):
        import numpy as np
        from PIL import Image
        if not len(images):
            return np.empty((0, 1))
        rows = []
        with self.torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = self.torch.stack([self.preprocess(Image.fromarray(np.asarray(image, dtype=np.uint8)))
                                          for image in images[start:start + self.batch_size]]).to(self.device)
                rows.append(self.forward_tensor(batch).float().cpu().numpy())
        return np.concatenate(rows)

    def detections(self, image):
        return []


def load_adapter(kind, args, device):
    if kind == 'OC_VGG':
        adapter = ClassifierAdapter(args, device, 'vgg16')
        adapter.model.requires_grad_(False)
        return adapter
    if kind in {'OC_PartImageNet', 'OC_PACO'}:
        adapter = ClassifierAdapter(args, device, 'resnet50')
        adapter.model.requires_grad_(False)
        return adapter
    if kind == 'OC_BUSI':
        from src.medical.classifier import BusiAdapter
        return BusiAdapter(args, device)
    if kind in {'OC_FunnyBirds', 'OC_KvasirSEG', 'OC_ISIC2018'}:
        from dataset_adapters import ADAPTERS
        return ADAPTERS[kind](args, device)
    if kind == 'OC_CelebA':
        from celeba_model import CelebaAdapter
        return CelebaAdapter(args, device)
    if kind == 'OD_DETR':
        return DetrAdapter(args, device)
    if kind == 'OD_DOTA':
        return YoloObbAdapter(args, device)
    if kind == 'VL_RefCOCO':
        return ClipAdapter(args, device)
    return ClassifierAdapter(args, device, 'resnet50' if kind in {'OC_ResNet', 'OC_PartImageNet'} else 'vit_b_16')
