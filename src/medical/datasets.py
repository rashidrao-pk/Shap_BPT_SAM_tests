"""BUSI images and expert lesion masks (annotations are never model inputs)."""
from pathlib import Path
import re
import numpy as np
from PIL import Image

MASK = re.compile(r"_mask(?:_\d+)?$", re.I)


def discover_images(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = []
    seen = set()
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp'} or not path.is_file() or MASK.search(path.stem):
            continue
        if path.stem in seen:
            raise ValueError(f'Duplicate BUSI image stem: {path.stem}')
        seen.add(path.stem)
        result.append({'id': path.stem, 'file_name': str(path.relative_to(root))})
    return result


def lesion_mask(image_path):
    """Union all numbered masks; preserve orientation and reject mismatched shapes."""
    path = Path(image_path)
    with Image.open(path) as image:
        shape = (image.height, image.width)
    pattern = re.compile(re.escape(path.stem) + r'_mask(?:_\d+)?$', re.I)
    paths = sorted(p for p in path.parent.iterdir()
                   if p.is_file() and p.suffix.lower() == '.png' and pattern.fullmatch(p.stem))
    union = np.zeros(shape, dtype=bool)
    for mask_path in paths:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert('L')) > 0
        if mask.shape != shape:
            raise ValueError(f'BUSI mask/image size mismatch: {mask_path}')
        union |= mask
    return union, paths
