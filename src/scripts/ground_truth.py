"""Dataset GT loaders for CAMO, CelebAMask-HQ, ImageNet-S, and COCO previews."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import re
import textwrap

import numpy as np
from PIL import Image
from scipy import ndimage

from config_paths import resolve_path

PARTS = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear',
         'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
IMAGENETS_MERGES = {'n04356056': 'n04355933', 'n04493381': 'n02808440',
                    'n03642806': 'n03832673', 'n04008634': 'n03773504', 'n03887697': 'n15075141'}


def title(gt):
    count = gt.get('display_count', gt.get('instance_count', 0))
    unit = gt.get('count_label', 'instances')
    return f"GT: {gt['class_name']}\n{count} {unit}; {textwrap.fill(gt['status'], 48)}"


def _result(labels, shape, name, paths, status='matched', selected=None, ignored=None, **extra):
    source_shape = list(labels.shape)
    if labels.shape != tuple(shape[:2]):
        labels = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(
            (shape[1], shape[0]), resample=Image.Resampling.NEAREST))
    ignored_mask = np.isin(labels, ignored or [])
    mask = np.isin(labels, selected) if selected is not None else (labels != 0)
    mask &= ~ignored_mask
    count = int(ndimage.label(mask)[1])
    return {'class_name': name, 'mask': mask, 'label_map': labels,
            'instance_count': None, 'display_count': count, 'count_label': 'components',
            'foreground_pixels': int(mask.sum()), 'status': status,
            'source_paths': [str(path) for path in paths], 'source_shape': source_shape,
            'resized': source_shape != list(shape[:2]), 'ignored_pixels': int(ignored_mask.sum()), **extra}


def _empty(shape, name, status, **extra):
    return _result(np.zeros(shape[:2], dtype=np.uint16), shape, name, [], status, **extra)


def validate_config(config):
    settings = config['data'].get('ground_truth')
    if not isinstance(settings, dict) or not settings.get('root') or not settings.get('format'):
        raise ValueError('Configure data.ground_truth.format and data.ground_truth.root for --show_gt')
    root = resolve_path(settings['root'])
    if not root.is_dir():
        raise FileNotFoundError(f'Ground-truth directory not found: {root}')
    return settings, root


@lru_cache(maxsize=8)
def _mask_index(root_text, format_name, split):
    root = Path(root_text)
    if format_name == 'imagenets':
        target = f'ImageNetS{split}'
        choices = ([root] if root.name.lower() == target.lower() else [root / target]
                   if (root / target).is_dir() else [path for path in root.rglob(target) if path.is_dir()])
        known = [path for path in [root, *root.parents] if re.fullmatch(r'ImageNetS\d+', path.name, re.I)]
        if known and known[0].name.lower() != target.lower():
            raise ValueError(f'GT root belongs to {known[0].name}, not {target}; change ground_truth.variant')
        if not choices and any(path.is_dir() and re.fullmatch(r'ImageNetS\d+', path.name, re.I)
                               for path in root.iterdir()):
            raise ValueError(f'{target} not found under {root}; configure the available ImageNetS variant')
        base = choices[0] if len(choices) == 1 else root
        segmentation_names = {'validation-segmentation', 'train-semi-segmentation', 'test-segmentation'}
        direct = [base / name for name in segmentation_names if (base / name).is_dir()]
        segmentation = ([base] if 'segmentation' in base.name.lower() else direct or
                        [path for path in base.rglob('*') if path.is_dir() and path.name in segmentation_names])
        folders = segmentation or [base]
    else:
        matching_split = [path for path in root.iterdir() if path.is_dir() and path.name.lower() == split.lower()]
        folders = matching_split or [root]
    indexed = {}
    for folder in folders:
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.png', '.bmp', '.tif', '.tiff'}:
                indexed.setdefault(path.stem, set()).add(path)
    return {key: sorted(paths) for key, paths in indexed.items()}


def _one_mask(root, format_name, split, stem):
    paths = _mask_index(str(root), format_name, str(split)).get(stem, [])
    if len(paths) > 1:
        raise ValueError(f'Ambiguous GT for {stem}: {paths}; narrow data.ground_truth.root/split')
    return paths[0] if paths else None


def _load_camo(settings, root, stem, shape):
    path = _one_mask(root, 'binary', settings.get('split', 'Train'), stem)
    if path is None:
        return _empty(shape, 'Camouflaged object', 'annotation not available', scope='dataset_foreground')
    with Image.open(path) as source:
        labels = np.asarray(source.convert('L')) > 0
    return _result(labels.astype(np.uint8), shape, 'Camouflaged object', [path], scope='dataset_foreground')


def _load_celeba(settings, root, stem, shape, gt_class):
    if not stem.isdigit():
        raise ValueError(f'CelebAMask-HQ expects a numeric image stem, got {stem}')
    index = int(stem)
    selected = settings.get('parts', PARTS) if gt_class in {None, 'all', 'foreground'} else [gt_class]
    if isinstance(selected, str):
        selected = [selected]
    unknown = set(selected) - set(PARTS)
    if unknown:
        raise ValueError(f'Unknown CelebAMask-HQ parts: {sorted(unknown)}; supported: {PARTS}')
    labels, paths, present = None, [], []
    for part in selected:
        path = root / str(index // 2000) / f'{index:05d}_{part}.png'
        if not path.is_file():
            continue
        with Image.open(path) as source:
            mask = np.asarray(source.convert('L')) > 0
        if labels is None:
            labels = np.zeros(mask.shape, dtype=np.uint16)
        if mask.shape != labels.shape:
            raise ValueError(f'CelebAMask-HQ part masks disagree in shape: {path}')
        labels[mask] = PARTS.index(part) + 1
        paths.append(path)
        present.append(part)
    name = ', '.join(selected) if len(selected) < 3 else 'CelebAMask-HQ face parts'
    if labels is None:
        return _empty(shape, name, 'annotation not available', scope='face_parts', part_names=selected)
    return _result(labels, shape, name, paths, scope='face_parts', part_names=present,
                   label_names={index + 1: part for index, part in enumerate(PARTS)},
                   display_count=len(present), count_label='parts')


@lru_cache(maxsize=8)
def _categories(root_text, split, explicit):
    if explicit:
        path = resolve_path(explicit)
    else:
        root = Path(root_text)
        names = [f'ImageNetS_categories_im{split}.txt', f'ImageNetS_categories_im{split}_sort.txt']
        candidates = [path for name in names for path in root.rglob(name)]
        path = candidates[0] if candidates else None
    if path is None:
        return None
    values = [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]
    if not all(re.fullmatch(r'n\d{8}', value) for value in values):
        raise ValueError(f'Expected a synset per line in ImageNetS categories: {path}')
    if len(set(values)) != len(values) or (str(split) in {'50', '300', '919'} and len(values) != int(split)):
        raise ValueError(f'ImageNetS category file does not match variant {split}: {path}')
    # ImageNetS label i+1 follows line i, as in the reference notebook.
    return values


@lru_cache(maxsize=4)
def _classifier_synsets(explicit):
    if explicit:
        path = resolve_path(explicit)
        if path.suffix.lower() == '.json':
            values = json.loads(path.read_text())
            ordered = values if isinstance(values, list) else [values[str(i)] for i in range(len(values))]
            return [value[0] if isinstance(value, list) else value for value in ordered]
        return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
    # Read timm's packaged metadata without importing torch/timm into the viewer.
    spec = importlib.util.find_spec('timm')
    if spec and spec.origin:
        path = Path(spec.origin).parent / 'data/_info/imagenet_synsets.txt'
        if path.is_file():
            return path.read_text().splitlines()
    return None


def _load_imagenets(settings, root, stem, shape, gt_class, predicted_index, label_space, predicted_class):
    split = str(settings.get('variant', 919))
    path = _one_mask(root, 'imagenets', split, stem)
    if path is None:
        return _empty(shape, 'ImageNetS', 'annotation not available', scope='unavailable')
    with Image.open(path) as source:
        encoded = np.asarray(source)
    if encoded.ndim == 3:
        labels = encoded[:, :, 0].astype(np.uint16) + 256 * encoded[:, :, 1].astype(np.uint16)
    elif encoded.ndim == 2:
        labels = encoded.astype(np.uint16)
    else:
        raise ValueError(f'Unexpected ImageNetS mask shape: {encoded.shape}')
    valid = [int(value) for value in np.unique(labels) if value not in {0, 1000}]
    categories = _categories(str(root), split, settings.get('categories_file'))
    label_names = {i + 1: synset for i, synset in enumerate(categories)} if categories else {}
    target = gt_class
    if target is None and predicted_index is not None:
        mapping = settings.get('class_index_file')
        synsets = _classifier_synsets(mapping) if mapping or label_space == 'imagenet1k' else None
        if synsets is None or not 0 <= predicted_index < len(synsets):
            return _result(labels, shape, 'ImageNetS predicted class', [path],
                           'class mapping unavailable; use --gt-class foreground or configure class_index_file',
                           selected=[], ignored=[1000], scope='mapping_unavailable', label_names=label_names)
        target = synsets[predicted_index]
    if target in {None, 'all', 'foreground'}:
        selected, name, scope = valid, 'ImageNetS annotated foreground', 'dataset_foreground'
    elif str(target).isdigit():
        selected, name, scope = [int(target)], f'ImageNetS label {target}', 'selected_class'
        if int(target) in {0, 1000}:
            raise ValueError('ImageNetS IDs 0 and 1000 are background/ignore, not object categories')
    else:
        if not re.fullmatch(r'n\d{8}', str(target)):
            raise ValueError('For ImageNetS, --gt-class must be a synset, numeric GT label ID, or foreground')
        target = IMAGENETS_MERGES.get(target, target) if split == '919' else target
        name = (f'{predicted_class} ({target})' if gt_class is None and predicted_class else f'ImageNetS {target}')
        scope = 'selected_class'
        if categories:
            selected = [categories.index(target) + 1] if target in categories else []
        else:
            parent_synset = next((parent.name for parent in path.parents
                                  if re.fullmatch(r'n\d{8}', parent.name)), None)
            if parent_synset and len(valid) == 1:
                selected = valid if target == parent_synset else []
            else:
                return _result(labels, shape, name, [path],
                               'ImageNetS category mapping unavailable; configure categories_file',
                               selected=[], ignored=[1000], scope='mapping_unavailable')
    present = sorted(set(selected) & set(valid))
    status = 'matched' if present else 'no annotated pixels for selected class'
    return _result(labels, shape, name, [path], status, selected=selected, ignored=[1000],
                   scope=scope, category_ids=selected, label_names=label_names,
                   display_count=len(present), count_label='classes')


@lru_cache(maxsize=2)
def _coco(annotation_file):
    from pycocotools.coco import COCO
    return COCO(annotation_file)


def load_ground_truth(config, image_path, image_shape, gt_class=None, predicted_index=None, label_space=None,
                      predicted_class=None):
    """Load dataset GT; never interpret RGB encoding as a grayscale class mask."""
    stem = Path(image_path).stem
    if config['data'].get('name', '').replace('_', '').lower() == 'mscoco':
        coco = _coco(str(resolve_path(config['data']['annotation_file'])))
        categories = coco.loadCats(coco.getCatIds())
        ids = [category['id'] for category in categories if gt_class in {None, 'all', 'foreground'}
               or category['name'].lower() == gt_class.lower()]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[int(stem)], catIds=ids)) if ids else []
        labels = np.zeros(image_shape[:2], dtype=np.uint16)
        used = []
        for annotation in annotations:
            if annotation.get('segmentation'):
                labels[np.asarray(coco.annToMask(annotation), dtype=bool)] = len(used) + 1
                used.append(annotation['id'])
        return _result(labels, image_shape, gt_class or 'COCO annotated instances',
                       [resolve_path(config['data']['annotation_file'])],
                       'matched' if used else 'no matching segmentation',
                       instance_count=len(used), display_count=len(used), count_label='instances',
                       annotation_ids=used, category_ids=ids, scope='selected_class' if gt_class else 'all_instances')
    settings, root = validate_config(config)
    format_name = settings['format'].lower()
    if format_name in {'binary', 'camo'}:
        if gt_class not in {None, 'all', 'foreground'}:
            raise ValueError('CAMO GT is binary; use --gt-class foreground')
        return _load_camo(settings, root, stem, image_shape)
    if format_name in {'celebamask_hq', 'celebahq'}:
        return _load_celeba(settings, root, stem, image_shape, gt_class)
    if format_name == 'imagenets':
        return _load_imagenets(settings, root, stem, image_shape, gt_class, predicted_index, label_space, predicted_class)
    raise ValueError(f'Unsupported GT format: {format_name}')


def save_ground_truth(gt, image, folder, image_id):
    import matplotlib.pyplot as plt
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mask_path = folder / f'{image_id}_gt.npy'
    labels_path = folder / f'{image_id}_gt_labels.npz'
    preview_path = folder / f'{image_id}_gt.png'
    np.save(mask_path, gt['mask'], allow_pickle=False)
    np.savez_compressed(labels_path, labels=gt['label_map'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        axes[0].imshow(gt['mask'], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"GT mask: {gt['class_name']}")
        axes[1].imshow(image)
        overlay = np.zeros((*gt['mask'].shape, 4))
        overlay[gt['mask']] = [0, 1, 0, .45]
        axes[1].imshow(overlay)
        axes[1].set_title(title(gt), fontsize=9)
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(preview_path, dpi=140)
    finally:
        plt.close(fig)
    metadata = {key: value for key, value in gt.items() if key not in {'mask', 'label_map'}}
    metadata.update(mask_path=str(mask_path), labels_path=str(labels_path), preview_path=str(preview_path))
    (folder / f'{image_id}_gt.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
