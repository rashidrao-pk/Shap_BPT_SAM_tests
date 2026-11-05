#!/usr/bin/env python3
"""One-time PartImageNet Seg/OOD archive and ResNet50 download; downloads use the standard library; optional configuration uses PyYAML."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import urllib.request
import zipfile

MODEL_URL = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'


def download(url, destination, hash_prefix=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        print(f'Existing: {destination}', flush=True)
    else:
        temporary = destination.with_suffix(destination.suffix + '.part')
        print(f'Downloading: {url}\nTo: {destination}', flush=True)
        urllib.request.urlretrieve(url, temporary)
        if hash_prefix:
            check_hash(temporary, hash_prefix)
        temporary.replace(destination)
    if hash_prefix:
        check_hash(destination, hash_prefix)
    return destination


def check_hash(path, prefix):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if not digest.hexdigest().startswith(prefix):
        raise ValueError(f'Checkpoint checksum mismatch: {path}; download a fresh copy')


def extract(archive, root):
    root = Path(root).resolve()
    if not zipfile.is_zipfile(archive):
        raise ValueError(f'Not a ZIP archive: {archive}; check for a login page or incomplete download')
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError(f'Unsafe ZIP path: {member.filename}')
        bad = zipped.testzip()
        if bad:
            raise ValueError(f'Damaged archive member: {bad}')
        # Preserve already extracted files to make repeated setup inexpensive.
        for member in zipped.infolist():
            if not (root / member.filename).exists():
                zipped.extract(member, root)


def configure_dataset(dataset_root, config_path, split='val', annotation_file=None, variant='Seg'):
    """Locate an extracted COCO part split and validate every image before writing paths."""
    import yaml
    from partimagenet_data import collect_images
    root = Path(dataset_root).resolve()
    config_path = Path(config_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root not found: {root}')
    if annotation_file:
        candidates = [Path(annotation_file).resolve()]
    else:
        names = {'val', 'validation'} if split == 'val' else {split}
        candidates = []
        for path in sorted(root.rglob('*.json')):
            relative = path.relative_to(root)
            tokens = {token.lower() for token in re.split(r'[/_.-]+', str(relative.with_suffix('')))}
            other_variant = 'partimagenet_ood' if variant == 'Seg' else 'partimagenet_seg'
            if any(other_variant in part.lower() for part in relative.parts):
                continue
            if names & tokens:
                candidates.append(path)
        valid = []
        for path in candidates:
            try:
                document = json.loads(path.read_text())
                if (isinstance(document, dict) and document.get('images') and document.get('categories')
                        and any(ann.get('segmentation') for ann in document.get('annotations', []))):
                    valid.append(path)
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        candidates = valid
    if len(candidates) != 1:
        found = '\n'.join(str(p) for p in candidates) or '(none)'
        raise ValueError(f'Expected one {split} COCO part annotation JSON, found {len(candidates)}:\n{found}\n'
                         'Inspect the extracted JSON files and pass --annotation-file /absolute/path/to/file.json')
    annotation = candidates[0]
    document = json.loads(annotation.read_text())
    if not (isinstance(document, dict) and document.get('images') and document.get('categories')
            and any(ann.get('segmentation') for ann in document.get('annotations', []))):
        raise ValueError(f'Not a COCO part-segmentation annotation JSON: {annotation}')
    records = collect_images({'image_dir': str(root), 'annotation_file': str(annotation)})
    if not records:
        raise ValueError(f'No images in selected annotations: {annotation}')
    # Keep the extraction root: the loader resolves nested image directories by filename.
    config = yaml.safe_load(config_path.read_text())
    data = config['data']
    if data.get('name', '').lower() != 'partimagenet':
        raise ValueError('The target config must be a PartImageNet config')
    data.update(image_dir=str(root), annotation_file=str(annotation), input_format='directory')
    data['ground_truth'] = {**data.get('ground_truth', {}), 'format': 'partimagenet',
                            'root': str(root), 'annotation_file': str(annotation)}
    backup = config_path.with_suffix(config_path.suffix + '.bak')
    if not backup.exists():
        backup.write_bytes(config_path.read_bytes())
    temporary = config_path.with_suffix(config_path.suffix + '.tmp')
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    temporary.replace(config_path)
    print(f'Updated config: {config_path}\nImages: {root}\nAnnotations: {annotation}\n'
          f'Validated split images: {len(records)}\nOriginal config backup: {backup}', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', choices=['all', 'dataset', 'model', 'configure'], default='all')
    parser.add_argument('--variant', choices=['Seg', 'OOD'], default='Seg')
    parser.add_argument('--dataset-root', type=Path, default=Path('/beegfs/home/mrashid/datasets/PartImageNet'))
    parser.add_argument('--model-path', type=Path, default=Path('checkpoints/partimagenet/resnet50-11ad3fa6.pth'))
    parser.add_argument('--configure', action='store_true', help='Configure extracted image/GT paths after downloads')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/PartImageNet_epito.yaml')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--annotation-file', type=Path, help='Explicit COCO part JSON if automatic discovery is ambiguous')
    args = parser.parse_args()
    if args.assets in {'all', 'dataset'}:
        filename = f'PartImageNet_{args.variant}.zip'
        archive = download(f'https://huggingface.co/datasets/turkeyju/PartImageNet/resolve/main/{filename}',
                           args.dataset_root / filename)
        print('Checking and extracting archive...', flush=True)
        extract(archive, args.dataset_root)
        print(f'Dataset extracted under {args.dataset_root}; configure image_dir and annotation_file for your split.')
    if args.assets in {'all', 'model'}:
        download(MODEL_URL, args.model_path, hash_prefix='11ad3fa6')
        print(f'Use --checkpoint {args.model_path} --weights IMAGENET1K_V2', flush=True)
    if args.configure or args.assets == 'configure':
        configure_dataset(args.dataset_root, args.config, args.split, args.annotation_file, args.variant)


if __name__ == '__main__':
    main()
