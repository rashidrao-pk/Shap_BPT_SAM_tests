"""Resolve only images listed by a COCO-style part annotation split (PartImageNet, PACO)."""
import json
from pathlib import Path


def collect_images(data):
    from config_paths import resolve_path
    dataset = data.get('name', 'PartImageNet')
    root = resolve_path(data['image_dir']).resolve()
    annotation = resolve_path(data['annotation_file'])
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = json.loads(annotation.read_text())['images']
    paths_by_name = None
    selected, seen = [], set()
    for item in records:
        name = Path(item['file_name'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError(f'Unsafe {dataset} image path: {name}')
        path = root / name
        if not path.is_file():
            if paths_by_name is None:
                paths_by_name = {}
                for candidate in root.rglob('*'):
                    if candidate.is_file():
                        paths_by_name.setdefault(candidate.name, []).append(candidate)
            matches = paths_by_name.get(name.name, [])
            if len(matches) != 1:
                raise ValueError(f'{name}: expected one matching image under {root}, found {len(matches)}')
            path = matches[0]
        if path.stem in seen:
            raise ValueError(f'Duplicate {dataset} image stem: {path.stem}')
        seen.add(path.stem)
        selected.append({'id': path.stem, 'file_name': str(path.relative_to(root)),
                         'annotation_id': item['id']})
    return sorted(selected, key=lambda item: item['id'])
