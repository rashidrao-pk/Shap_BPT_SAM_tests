"""Index ground-truth filenames without decoding masks or importing NumPy."""
from functools import lru_cache
from pathlib import Path
import re


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
