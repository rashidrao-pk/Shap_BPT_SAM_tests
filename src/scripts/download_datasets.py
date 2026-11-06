#!/usr/bin/env python3
"""One-time download of FunnyBirds, PACO-LVIS, Kvasir-SEG, ISIC 2018 and their frozen classifiers.

Files are pinned by URL and SHA-256; Hugging Face files also by commit, so a repository
update cannot silently change the explained model. Only ISIC preparation needs Pillow.
"""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import shutil
import ssl
import urllib.error
import urllib.request

from download_partimagenet import MODEL_URL, check_hash, download, extract

DATASET_ROOT = Path('/beegfs/home/mrashid/datasets')
VISINF = 'https://download.visinf.tu-darmstadt.de/data/funnybirds'
ISIC = 'https://isic-archive.s3.amazonaws.com/challenges/2018'
ISIC_MODEL = ('https://huggingface.co/adeIaide/skin-lesion-7class-efficientnet/resolve/'
              'b5ebe73c3b89baa90ee4f991afa318385414a973')
KVASIR_MODEL = ('https://huggingface.co/mmuratarat/kvasir-v2-classifier/resolve/'
                '4125333063ba077e18b1f9ef40df2de5fc50453e')
PACO_ZIP_SHA256 = '02ac4edb22c251e07853e6231d69aec3fad0a180f03de2f8c880650322debc80'
KVASIR_ZIP_SHA256 = '03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7'
# (URL, path below the repository, SHA-256)
MODELS = {
    'funnybirds': [(f'{VISINF}/models/resnet50_final_0_checkpoint_best.pth.tar',
                    'checkpoints/funnybirds/resnet50_final_0_checkpoint_best.pth.tar',
                    '88f0fcf517ab5aa318325db7ffef46d65b52f0b8e18c325f5280d97f631ad139')],
    # PACO shares PartImageNet's frozen ImageNet classifier.
    'paco': [(MODEL_URL, 'checkpoints/partimagenet/resnet50-11ad3fa6.pth', '11ad3fa6')],
    'kvasirseg': [(f'{KVASIR_MODEL}/config.json', 'checkpoints/kvasir_vit/config.json', 'b620881f06b0f6385943ed80ea85d3feb2a049c89f4a3626ac194fe95cc1bb86'),
                  (f'{KVASIR_MODEL}/preprocessor_config.json', 'checkpoints/kvasir_vit/preprocessor_config.json', 'b09d2030f83f2a59d12c717d41a9135a3f0c1ba0a2a5df694dbc40f77735daed'),
                  (f'{KVASIR_MODEL}/pytorch_model.bin', 'checkpoints/kvasir_vit/pytorch_model.bin',
                   '2fd2526a95c9f4ae4667982d55065588aefc861114f6134e4d8954e0d82a5256')],
    'isic2018': [(f'{ISIC_MODEL}/config.json', 'checkpoints/isic2018_efficientnet_b0/config.json', '3b09f5eb750b0d458f208b4c7c46e3babe58a46106f3fb0e9eac8d3a1d88f3f8'),
                 (f'{ISIC_MODEL}/model.safetensors', 'checkpoints/isic2018_efficientnet_b0/model.safetensors',
                  'fed0c3cb0a15823f4f6a82d4dd2725a0b8e6068a971664e1504403fba5b54562')],
}


def download_funnybirds(root, args):
    archive = download(f'{VISINF}/FunnyBirds.zip', root / 'FunnyBirds/FunnyBirds.zip')
    print('Checking and extracting archive...', flush=True)
    extract(archive, root)


def download_pinned(url, destination, sha256):
    """Fetch one file whose full SHA-256 is pinned, tolerating a server's incomplete TLS chain.

    Simula's server omits its intermediate certificate, which Python rejects. The pinned hash,
    not TLS, then establishes integrity: unverified bytes never reach the destination.
    """
    try:
        return download(url, destination, hash_prefix=sha256)
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
    print(f'TLS chain rejected for {url}; retrying, with the pinned SHA-256 as the integrity check', flush=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    with urllib.request.urlopen(url, context=ssl._create_unverified_context()) as response, temporary.open('wb') as stream:
        shutil.copyfileobj(response, stream)
    try:
        check_hash(temporary, sha256)
    except ValueError:
        temporary.unlink()
        raise
    temporary.replace(destination)
    return destination


def download_kvasirseg(root, args):
    archive = download_pinned('https://datasets.simula.no/downloads/kvasir-seg.zip',
                              root / 'Kvasir-SEG/kvasir-seg.zip', KVASIR_ZIP_SHA256)
    extract(archive, root)


def fetch_image(record, image_root):
    """Fetch one COCO image, keeping the file_name layout that PACO annotations use."""
    path = image_root / record['file_name']
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(record['coco_url'], timeout=60) as response:
        content = response.read()
    if not content.startswith(b'\xff\xd8'):
        raise ValueError(f"Not a JPEG response: {record['coco_url']}")
    temporary = path.with_suffix('.part')
    temporary.write_bytes(content)
    temporary.replace(path)


def download_paco(root, args):
    folder = root / 'PACO'
    archive = download('https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip',
                       folder / 'annotations/paco_lvis_v1.zip', hash_prefix=PACO_ZIP_SHA256)
    extract(archive, folder / 'annotations')
    annotation = folder / f'annotations/paco_lvis_v1_{args.split}.json'
    records = json.loads(annotation.read_text())['images']
    print(f'Fetching {len(records)} {args.split} images listed by {annotation.name}...', flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        tasks = {pool.submit(fetch_image, record, folder / 'images'): record for record in records}
        failed = [(tasks[task]['file_name'], task.exception())
                  for task in concurrent.futures.as_completed(tasks) if task.exception()]
    if failed:
        raise RuntimeError(f'{len(failed)} images failed; rerun to retry them. First: {failed[0]}')


def prepare_isic(images, masks, output, max_side):
    """Downscale images and masks together: the classifier input is only 320 pixels."""
    from PIL import Image
    (output / 'images').mkdir(parents=True, exist_ok=True)
    (output / 'masks').mkdir(parents=True, exist_ok=True)
    paths = sorted(images.glob('ISIC_*.jpg'))
    if not paths:
        raise FileNotFoundError(f'No ISIC images in {images}')
    for path in paths:
        mask_path = masks / f'{path.stem}_segmentation.png'
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        image_target = output / 'images' / f'{path.stem}.png'
        mask_target = output / 'masks' / mask_path.name
        if image_target.is_file() and mask_target.is_file():
            continue
        with Image.open(path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(f'ISIC mask/image size mismatch: {mask_path}')
            scale = min(1, max_side / max(image.size))
            size = tuple(max(1, round(side * scale)) for side in image.size)
            image.convert('RGB').resize(size, Image.Resampling.LANCZOS).save(image_target)
            mask.convert('L').resize(size, Image.Resampling.NEAREST).save(mask_target)
    return len(paths)


def task3_overlap(task1_ids, task3_csv):
    """Task 1 images that a Task 3 classifier could have trained on, by ISIC image ID."""
    with Path(task3_csv).open(newline='') as stream:
        return sorted(set(task1_ids) & {row['image'] for row in csv.DictReader(stream)})


def download_isic2018(root, args):
    folder = root / 'ISIC2018'
    name = args.split.capitalize() if args.split != 'train' else 'Training'
    extracted = {}
    for label, stem in [('images', f'ISIC2018_Task1-2_{name}_Input'), ('masks', f'ISIC2018_Task1_{name}_GroundTruth'),
                        ('task3', 'ISIC2018_Task3_Training_GroundTruth')]:
        archive = download(f'{ISIC}/{stem}.zip', folder / 'raw' / f'{stem}.zip')
        extract(archive, folder / 'raw')
        extracted[label] = folder / 'raw' / stem
    output = folder / f'prepared_{args.max_side}' / args.split
    count = prepare_isic(extracted['images'], extracted['masks'], output, args.max_side)
    overlap = task3_overlap([path.stem for path in extracted['images'].glob('ISIC_*.jpg')],
                            extracted['task3'] / 'ISIC2018_Task3_Training_GroundTruth.csv')
    print(f'Prepared {count} image/mask pairs: {output}\n'
          f'Task 1 {args.split} images also in Task 3 training, by ID: {len(overlap)}', flush=True)
    if overlap:
        print('Exclude before reporting held-out localization:', *overlap[:20], flush=True)


DATASETS = {'funnybirds': download_funnybirds, 'paco': download_paco,
            'kvasirseg': download_kvasirseg, 'isic2018': download_isic2018}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=[*DATASETS, 'all'])
    parser.add_argument('--assets', choices=['all', 'dataset', 'model'], default='all')
    parser.add_argument('--root', type=Path, default=DATASET_ROOT, help='Parent folder of all datasets')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2],
                        help='Repository holding checkpoints/')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val',
                        help='PACO annotation split or ISIC Task 1 split (default: val)')
    parser.add_argument('--max-side', type=int, default=1024, help='ISIC prepared image size limit')
    parser.add_argument('--workers', type=int, default=8, help='Parallel PACO image downloads')
    args = parser.parse_args()
    if args.split == 'val':
        args.split = 'validation' if args.dataset == 'isic2018' else 'val'
    for name in DATASETS if args.dataset == 'all' else [args.dataset]:
        if args.assets in {'all', 'dataset'}:
            DATASETS[name](args.root, args)
        if args.assets in {'all', 'model'}:
            for url, relative, digest in MODELS[name]:
                download(url, args.repo / relative, hash_prefix=digest)


if __name__ == '__main__':
    main()
