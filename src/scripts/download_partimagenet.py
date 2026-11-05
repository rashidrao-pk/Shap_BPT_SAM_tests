#!/usr/bin/env python3
"""One-time PartImageNet Seg/OOD archive and ResNet50 download; standard library only."""
import argparse
import hashlib
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', choices=['all', 'dataset', 'model'], default='all')
    parser.add_argument('--variant', choices=['Seg', 'OOD'], default='Seg')
    parser.add_argument('--dataset-root', type=Path, default=Path('/beegfs/home/mrashid/datasets/PartImageNet'))
    parser.add_argument('--model-path', type=Path, default=Path('checkpoints/partimagenet/resnet50-11ad3fa6.pth'))
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


if __name__ == '__main__':
    main()
