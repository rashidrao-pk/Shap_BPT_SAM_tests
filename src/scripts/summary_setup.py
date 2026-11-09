#!/usr/bin/env python3
"""Check that datasets, checkpoints, and packages are in place, then print summary tables.

Read-only: it checks files and package availability, without importing PyTorch, loading a
model, or using a GPU. Run it on the kind of node that runs the jobs: packages can be built
for a different architecture than the login node. Add --verify-hash to also read checkpoints
and compare their SHA-256 where a pinned value exists.
"""
import argparse
from collections import namedtuple
from dataclasses import dataclass
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re

from check_partition_progress import DEFAULT_CONFIGS, ROOT, load_config, rooted, source_ids
from config_paths import output_path
from gt_mask_index import _mask_index

OK, WARN, MISSING = 'OK', 'WARN', 'MISSING'
ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError)
Check = namedtuple('Check', 'status text detail')
ModelResult = namedtuple('ModelResult', 'model status location size detail')


@dataclass(frozen=True)
class Model:
    label: str
    doc: str
    kind: str                # file, folder, torch_cache, or hf_cache
    target: tuple            # Candidate paths, or the cache filename/repository
    size: int = None         # Exact bytes, to detect a truncated download
    files: tuple = ()        # Files required inside a folder
    packages: tuple = ()     # Needed to run this model


@dataclass(frozen=True)
class Dataset:
    label: str
    doc: str
    images: int = None       # For the split selected by the default config
    gt: int = None           # Images with annotations, when fewer than all
    sam2: str = 'sam2_tiny'
    models: tuple = ()
    packages: tuple = ()


MODELS = {
    'sam1': Model('SAM-1 ViT-B', 'SAM1.md', 'file', ('checkpoints/sam1/sam_vit_b_01ec64.pth',), 375042383),
    'sam2_tiny': Model('SAM-2 Tiny', 'SAM2.md', 'file', ('sam2.1_hiera_tiny.pt',), 156008466),
    'sam2_large': Model('SAM-2 Large', 'SAM2.md', 'file', ('sam2.1_hiera_large.pt',), 898083611),
    'yolo': Model('YOLO11s', 'YOLO.md', 'file',
                  ('src/notebooks/checkpoints/yolo11s.pt', 'src/checkpoints/yolo11s.pt', 'checkpoints/yolo11s.pt'),
                  19313732, packages=('ultralytics',)),
    'detr': Model('DETR-ResNet-50', 'DETR.md', 'hf_cache', ('facebook/detr-resnet-50',),
                  packages=('transformers', 'timm')),
    'resnet50_cache': Model('ResNet-50 (torch cache)', 'RESNET50.md', 'torch_cache',
                            ('resnet50-11ad3fa6.pth',), 102540417),
    'resnet50_file': Model('ResNet-50 (local file)', 'RESNET50.md', 'file',
                           ('checkpoints/partimagenet/resnet50-11ad3fa6.pth',), 102540417),
    'vgg16': Model('VGG16', 'VGG16.md', 'torch_cache', ('vgg16-397923af.pth',), 553433881),
    'vit_b16': Model('ViT-B/16', 'VIT_B16.md', 'torch_cache', ('vit_b_16-c867db91.pth',), 346328529),
    'celeba': Model('CelebA CNN', 'CELEBA_CNN.md', 'file', ('src/checkpoints/celeba_model.pth',), 41882800),
    'busi': Model('BUSI SigLIP', 'BUSI_SIGLIP.md', 'folder', ('checkpoints/busi_siglip',),
                  files=('config.json', 'model.safetensors'), packages=('transformers', 'safetensors')),
    'funnybirds': Model('FunnyBirds ResNet-50', 'FUNNYBIRDS_RESNET50.md', 'file',
                        ('checkpoints/funnybirds/resnet50_final_0_checkpoint_best.pth.tar',), 189249661),
    'kvasir': Model('Kvasir v2 ViT', 'KVASIR_VIT.md', 'folder', ('checkpoints/kvasir_vit',),
                    files=('config.json', 'pytorch_model.bin'), packages=('transformers',)),
    'isic': Model('ISIC EfficientNet-B0', 'ISIC_EFFICIENTNET.md', 'folder', ('checkpoints/isic2018_efficientnet_b0',),
                  files=('config.json', 'model.safetensors'), packages=('safetensors',)),
    'clip_b32': Model('CLIP ViT-B/32', 'CLIP.md', 'folder', ('checkpoints/clip_vit_b32',),
                      files=('config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt'), packages=('transformers',)),
    'yolo_obb': Model('YOLO11s-OBB', 'YOLO_OBB.md', 'file',
                      ('src/checkpoints/yolo11s-obb.pt', 'src/notebooks/checkpoints/yolo11s-obb.pt', 'checkpoints/yolo11s-obb.pt'),
                      19934534, packages=('ultralytics',)),
}

# Keyed by data.name without underscores, in lower case.
DATASETS = {
    'mscoco': Dataset('MS-COCO', 'MS_COCO.md', 5000, 4952, 'sam2_large', ('yolo', 'detr'), ('pycocotools',)),
    'imagenet': Dataset('ImageNet', 'ImageNet.md', 50000, 752, models=('resnet50_cache', 'vgg16', 'vit_b16')),
    'celebahq': Dataset('CelebA-HQ', 'CelebAHQ.md', 30000, models=('celeba',)),
    'camo': Dataset('CAMO', 'CAMO.md', 1000, models=('vit_b16',)),
    'busi': Dataset('BUSI', 'BUSI.md', 780, models=('busi',)),
    # One val image, n02134418_4463, has no annotation in the official JSON.
    'partimagenet': Dataset('PartImageNet', 'PARTIMAGENET.md', 1206, 1205, models=('resnet50_file',),
                            packages=('pycocotools',)),
    'funnybirds': Dataset('FunnyBirds', 'FUNNY_BIRD.md', 500, models=('funnybirds',)),
    'paco': Dataset('PACO', 'PACO.md', 2410, models=('resnet50_file',), packages=('pycocotools',)),
    'kvasirseg': Dataset('Kvasir-SEG', 'KVASIR_SEG.md', 1000, models=('kvasir',)),
    'isic2018': Dataset('ISIC 2018', 'ISIC_2018.md', 100, models=('isic',)),
    # Tiles selected by download_datasets.py --dataset dota (default --max-tiles 300).
    'dota': Dataset('DOTA', 'DOTA.md', 300, models=('yolo_obb',)),
    # RefCOCO (UNC) val: 1,500 COCO train2014 images, 3,811 referred objects, 10,834 expressions.
    'refcoco': Dataset('RefCOCO', 'REFCOCO.md', 1500, models=('clip_b32',), packages=('pycocotools',)),
}

COMMON = ('numpy', 'PIL', 'yaml', 'scipy', 'tqdm', 'torch', 'torchvision')
PARTITION_PACKAGES = (*COMMON, 'cv2')
XAI_PACKAGES = (*COMMON, 'matplotlib', 'pandas', 'skimage')
OPTIONAL_PACKAGES = ('captum', 'lime', 'shap')  # Needed only for --explain_all.
IMPORT_NAMES = {'PIL': 'Pillow', 'yaml': 'PyYAML', 'cv2': 'opencv-python', 'skimage': 'scikit-image'}


def worst(*statuses):
    return next((status for status in [MISSING, WARN] if status in statuses), OK)


def format_size(size):
    if size is None:
        return '-'
    value = float(size)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if value < 1000 or unit == 'GB':
            return f'{value:.0f} {unit}' if unit != 'GB' else f'{value:.1f} GB'
        value /= 1000


def table(headers, rows):
    """Plain aligned table, in the same spirit as check_partition_progress.py."""
    widths = [max(len(str(row[index])) for row in [headers, *rows]) for index in range(len(headers))]
    line = lambda row: '  '.join(str(value).ljust(width) for value, width in zip(row, widths)).rstrip()
    return '\n'.join([line(headers), '  '.join('-' * width for width in widths), *map(line, rows)])


def sam2_roots():
    values = [os.environ.get('SAM2_DIR'), ROOT / 'checkpoints/third_party/sam2',
              '/beegfs/home/mrashid/repos/third_party/sam2']
    return [Path(value).expanduser() / 'checkpoints' for value in values if value]


def torch_cache():
    home = os.environ.get('TORCH_HOME') or Path(os.environ.get('XDG_CACHE_HOME', '~/.cache')).expanduser() / 'torch'
    return Path(home).expanduser() / 'hub/checkpoints'


def hf_cache():
    home = os.environ.get('HF_HOME') or Path(os.environ.get('XDG_CACHE_HOME', '~/.cache')).expanduser() / 'huggingface'
    return Path(os.environ.get('HF_HUB_CACHE') or Path(home).expanduser() / 'hub').expanduser()


def candidates(model):
    if model.kind == 'torch_cache':
        return [torch_cache() / model.target[0]]
    if model.label.startswith('SAM-2'):
        return [root / model.target[0] for root in sam2_roots()]
    return [rooted(path) for path in model.target]


def hf_snapshot(repository):
    """A cached repository counts only with weights: an interrupted download leaves config files."""
    folder = hf_cache() / f"models--{repository.replace('/', '--')}/snapshots"
    for snapshot in sorted(folder.glob('*')) if folder.is_dir() else []:
        files = [path for path in snapshot.iterdir() if path.suffix in {'.safetensors', '.bin'}]
        if files and all(path.exists() for path in files) and (snapshot / 'config.json').exists():
            return snapshot, sum(path.stat().st_size for path in files)
    return None, None


def pinned_hashes():
    """Pinned SHA-256 values by repository-relative path; the download scripts stay the source."""
    from download_celeba_checkpoint import SHA256
    from download_datasets import MODELS as DOWNLOADS
    pins = {relative: digest for files in DOWNLOADS.values() for _, relative, digest in files}
    pins['src/checkpoints/celeba_model.pth'] = SHA256
    return pins


def verify_hash(path, pins):
    from download_partimagenet import check_hash
    relative = str(Path(path).resolve().relative_to(ROOT.resolve())) if Path(path).resolve().is_relative_to(ROOT.resolve()) else None
    if relative not in pins:
        return None
    try:
        check_hash(path, pins[relative])
    except ValueError:
        return 'SHA-256 differs from the pinned value'
    return None


def check_model(model, override=None, pins=None):
    """Find one model; override is the config's model.checkpoint, which the runners read first."""
    if model.kind == 'hf_cache':
        snapshot, size = hf_snapshot(model.target[0])
        if snapshot is None:
            return ModelResult(model, MISSING, str(hf_cache()), None, 'not in the Hugging Face cache, or weights incomplete')
        return ModelResult(model, OK, str(snapshot), size, '')
    paths = [rooted(override)] if override else candidates(model)
    found = next((path for path in paths if path.exists()), None)
    if found is None:
        return ModelResult(model, MISSING, str(paths[0]), None, 'not found')
    if model.kind == 'folder':
        absent = [name for name in model.files if not (found / name).is_file()]
        if absent:
            return ModelResult(model, MISSING, str(found), None, f"missing {', '.join(absent)}")
        size = sum((found / name).stat().st_size for name in model.files)
        problems = [verify_hash(found / name, pins) for name in model.files] if pins else []
    else:
        size = found.stat().st_size
        if model.size and size != model.size:
            return ModelResult(model, MISSING, str(found), size,
                               f'size differs from {model.size} bytes: incomplete or wrong file')
        problems = [verify_hash(found, pins)] if pins else []
    problem = next((item for item in problems if item), None)
    return ModelResult(model, MISSING if problem else OK, str(found), size, problem or '')


def shap_bpt_extension():
    """The compiled module must match this Python version, or import fails on the GPU node."""
    folder = ROOT / 'shap_bpt'
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    if any((folder / f'bpt{suffix}').is_file() for suffix in suffixes):
        return Check(OK, 'built', '')
    return Check(MISSING, 'not built', f'run: python setup.py build_ext --inplace (needs {suffixes[0]})')


def package_status(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def stems(paths):
    return {path.stem for path in paths if path.is_file()}


def coco_annotated(path):
    document = json.loads(Path(path).read_text())
    annotated = {annotation['image_id'] for annotation in document['annotations']}
    return {Path(image['file_name']).stem for image in document['images'] if image['id'] in annotated}


def masks_found(data):
    """Image stems that have ground truth, without decoding any mask."""
    settings = data.get('ground_truth')
    if settings is None:  # MS-COCO keeps its annotations at the top level.
        return coco_annotated(rooted(data['annotation_file']))
    format_name = settings['format'].lower()
    if format_name in {'partimagenet', 'paco'}:
        if rooted(settings['annotation_file']) != rooted(data['annotation_file']):
            raise ValueError('data.annotation_file and ground_truth.annotation_file select different splits')
        return coco_annotated(rooted(settings['annotation_file']))
    root = rooted(settings['root'])
    if not root.is_dir():
        raise FileNotFoundError(f'Ground-truth directory not found: {root}')
    if format_name in {'binary', 'camo'}:
        return set(_mask_index(str(root), 'binary', str(settings.get('split', 'Train'))))
    if format_name == 'imagenets':
        return set(_mask_index(str(root), 'imagenets', str(settings.get('variant', 919))))
    if format_name in {'celebamask_hq', 'celebahq'}:
        return {str(int(match.group(1))) for path in root.glob('*/*.png')
                if (match := re.match(r'(\d{5})_', path.name))}
    if format_name == 'funnybirds':
        return stems((root / f"{settings.get('split', 'test')}_part_map").rglob('*.png'))
    if format_name == 'kvasir_seg':
        return stems(root.glob('*.jpg'))
    if format_name == 'isic2018':
        return {path.name.removesuffix('_segmentation.png') for path in root.glob('*_segmentation.png')}
    if format_name == 'dota':
        return stems(root.glob('*.txt'))
    if format_name == 'refcoco':
        from refcoco_data import load_expressions
        return {entry['image_stem'] for entry in load_expressions(rooted(settings['expressions']))}
    if format_name == 'busi':
        return {re.sub(r'_mask(?:_\d+)?$', '', path.stem) for path in root.rglob('*_mask*.png')}
    raise ValueError(f'Unsupported GT format: {format_name}')


def check_images(spec, ids):
    count = len(ids)
    if not count:
        return Check(MISSING, '0', 'no images found')
    if spec.images and count != spec.images:
        return Check(WARN, f'{count}/{spec.images}', f'expected {spec.images} for the default split')
    return Check(OK, str(count), '')


def check_gt(spec, data, ids):
    try:
        found = masks_found(data) & ids
    except ERRORS as error:
        return Check(MISSING, 'none', str(error))
    # An ImageNet-S count applies only to the variant the default config selects.
    settings = data.get('ground_truth') or {}
    expected = spec.gt if str(settings.get('variant', 50)) == '50' else None
    expected = expected or len(ids)
    text = f'{len(found)}/{len(ids)}'
    if not found:
        return Check(MISSING, text, 'no image has ground truth')
    if len(found) < expected:
        return Check(WARN, text, f'{expected - len(found)} images have no ground truth')
    return Check(OK, text, '')


def completed_partitions(folder, ids):
    """Count XAI inputs from one listing: stat calls are slow on a network filesystem.

    check_partition_progress.py reads every metadata file, for exact progress and timing.
    """
    names = set(os.listdir(folder))
    return sum(f'{image_id}_compact.npy' in names and f'{image_id}_results.json' in names for image_id in ids)


def partition_folder(base, prefix, sam, configured):
    """Mirror the budget study's selection: one compact-sam folder for each SAM version."""
    folders = sorted(path for path in base.glob(f'{prefix}_{sam}_*compact-sam*') if path.is_dir()) if base.is_dir() else []
    chosen = [path for path in folders if path.name == configured]
    return (chosen or folders or [None])[0], len(folders)


def check_partitions(config, ids, sam):
    data = config['data']
    base = output_path(config, 'partitions')
    prefix = re.sub(r'[^A-Za-z0-9-]', '', data['name'])
    folder, matches = partition_folder(base, prefix, sam, data.get('mask_dir_final'))
    if folder is None:
        return Check(WARN, 'none', ''), 0
    completed = completed_partitions(folder, ids)
    detail = f'{matches} {sam} partition folders: select one with data.mask_dir_final' if matches > 1 else ''
    return Check(OK if completed == len(ids) and not detail else WARN, f'{completed}/{len(ids)}', detail), completed


def check_dataset(config_name, pins=None):
    """Collect every check for one dataset; a broken dataset must not stop the others."""
    result = {'config': config_name, 'problems': []}
    try:
        config = load_config(config_name)
        data = config['data']
        key = data['name'].replace('_', '').lower()
        spec = DATASETS[key]
    except ERRORS as error:
        result.update(label=str(config_name), error=str(error), images=Check(MISSING, 'bad config', str(error)))
        return result
    result.update(label=spec.label, spec=spec)
    problem = lambda area, check, doc: result['problems'].append((area, check.status, check.detail, doc))
    try:
        ids = source_ids(data)
        images = check_images(spec, ids)
    except FileNotFoundError as error:
        ids, images = set(), Check(MISSING, 'none', f'not found: {error.filename or error}')
    except ERRORS as error:
        ids, images = set(), Check(MISSING, 'none', str(error))
    # Without images, a ground-truth line would only repeat the same problem.
    gt = check_gt(spec, data, ids) if ids else Check(MISSING, 'none', '')
    result.update(images=images, gt=gt)
    for area, check in [('images', images), ('ground truth', gt)]:
        if check.status != OK and check.detail:
            problem(area, check, f'docs/DATASETS/{spec.doc}')
    sam = {'sam1': check_model(MODELS['sam1'], pins=pins), 'sam2': check_model(MODELS[spec.sam2], pins=pins)}
    override = config.get('model', {}).get('checkpoint')
    models = [check_model(MODELS[key], override if len(spec.models) == 1 else None, pins) for key in spec.models]
    result.update(sam=sam, models=models)
    selected = (source_ids(data, gt_only=True) if key == 'imagenet' and ids and gt.status != MISSING else ids)
    try:
        progress = {version: check_partitions(config, selected, version) for version in sam}
    except ERRORS as error:
        progress = {version: (Check(MISSING, 'error', str(error)), 0) for version in sam}
    result['partitions'] = {version: value[0] for version, value in progress.items()}
    for version, (check, _) in progress.items():
        if check.detail:
            problem(f'{version} partitions', check, 'docs/Partitions/DATASETS/README.md')
    configured = output_path(config, 'partitions') / data.get('mask_dir_final', '__unset__')
    result['configured_ready'] = completed_partitions(configured, selected) if configured.is_dir() else 0
    return result


def readiness(result, packages):
    """Separate 'cannot run' from 'not generated yet', which is the normal next step."""
    if 'spec' not in result:
        return 'BLOCKED', 'BLOCKED'
    spec = result['spec']
    data_ok = result['images'].status != MISSING
    has_sam = {'sam1': result['sam']['sam1'].status == OK and packages.get('segment_anything'),
               'sam2': result['sam']['sam2'].status == OK and packages.get('sam2')}
    partition = data_ok and any(has_sam.values()) and all(packages.get(name) for name in PARTITION_PACKAGES)
    need_xai = [*XAI_PACKAGES, *spec.packages, *[name for model in result['models'] for name in model.model.packages]]
    runnable = (data_ok and all(model.status == OK for model in result['models'])
                and all(packages.get(name) for name in need_xai) and packages.get('shap_bpt'))
    xai = 'BLOCKED' if not runnable else 'READY' if result['configured_ready'] else 'NO PARTS'
    return 'READY' if partition else 'BLOCKED', xai


def collect_packages(results):
    names = {*PARTITION_PACKAGES, *XAI_PACKAGES, *OPTIONAL_PACKAGES, 'segment_anything', 'sam2'}
    for result in results:
        if 'spec' in result:
            names.update(result['spec'].packages)
            names.update(name for model in result['models'] for name in model.model.packages)
    return {name: package_status(name) for name in sorted(names)}


def model_summary(results):
    """One row for each checkpoint, merged across the datasets that share it."""
    rows = {}
    for result in results:
        if 'spec' not in result:
            continue
        for item in [*result['sam'].values(), *result['models']]:
            key = (item.model.label, item.location)
            rows.setdefault(key, {'item': item, 'used_by': []})['used_by'].append(result['label'])
    return list(rows.values())


def build_report(config_names, verify=False):
    pins = pinned_hashes() if verify else None
    results = [check_dataset(name, pins) for name in config_names]
    packages = collect_packages(results)
    extension = shap_bpt_extension()
    packages['shap_bpt'] = extension.status == OK
    for result in results:
        result['can_partition'], result['can_explain'] = readiness(result, packages)
    return results, packages, extension


def dataset_row(result):
    if 'spec' not in result:
        return [result['label'], 'bad config', '-', '-', '-', '-', '-', '-', 'BLOCKED', 'BLOCKED']
    models = worst(*[item.status for item in result['models']])
    sam = result['sam']
    return [result['label'], f"{result['images'].status} {result['images'].text}",
            f"{result['gt'].status} {result['gt'].text}", sam['sam1'].status, sam['sam2'].status, models,
            result['partitions']['sam1'].text, result['partitions']['sam2'].text,
            result['can_partition'], result['can_explain']]


def print_report(results, packages, extension):
    print('\nDatasets\n')
    print(table(['Dataset', 'Images', 'Ground truth', 'SAM-1', 'SAM-2', 'Models', 'Parts SAM-1', 'Parts SAM-2',
                 'Partitions', 'XAI'], [dataset_row(result) for result in results]))
    print('\nCheckpoints\n')
    print(table(['Model', 'Status', 'Size', 'Used by', 'Location'],
                [[row['item'].model.label, row['item'].status, format_size(row['item'].size),
                  ', '.join(row['used_by']) if len(row['used_by']) < 4 else f"{len(row['used_by'])} datasets",
                  row['item'].location] for row in model_summary(results)]))
    print('\nPackages\n')
    rows = [[name, OK if found else (WARN if name in OPTIONAL_PACKAGES else MISSING),
             '--explain_all only' if name in OPTIONAL_PACKAGES else ''] for name, found in packages.items()
            if name != 'shap_bpt']
    rows.append(['shap_bpt extension', extension.status, extension.detail])
    print(table(['Package', 'Status', 'Note'], rows))
    print('\nPartitions: READY means generation can run. '
          'XAI: READY means it can run now; NO PARTS means generate partitions first.')
    problems = problem_lines(results, packages, extension)
    if problems:
        print('\nTo fix\n')
        print('\n'.join(problems))
    print(f"\nReady: {sum(result['can_partition'] == 'READY' for result in results)}/{len(results)} for partitions, "
          f"{sum(result['can_explain'] == 'READY' for result in results)}/{len(results)} for XAI.")


def problem_lines(results, packages, extension):
    lines = []
    for result in results:
        if 'spec' not in result:
            lines.append(f"- {result['label']}: {result.get('error', 'bad config')}")
        for area, status, detail, doc in result.get('problems', []):
            lines.append(f"- {result['label']} {area} [{status}]: {detail} (see {doc})")
    for row in model_summary(results):
        item = row['item']
        if item.status != OK:
            lines.append(f"- {item.model.label} [{item.status}]: {item.detail} (see docs/MODELS/{item.model.doc})")
    absent = [IMPORT_NAMES.get(name, name) for name, found in packages.items()
              if not found and name not in {*OPTIONAL_PACKAGES, 'shap_bpt'}]
    if absent:
        lines.append(f"- Packages not installed: {', '.join(absent)}")
    if extension.status != OK:
        lines.append(f'- shap_bpt: {extension.detail}')
    if absent or extension.status != OK:
        lines.append('  On a login node, check on a GPU node first: packages there can be built for another architecture.')
    return lines


def report_json(results, packages):
    rows = []
    for result in results:
        spec = 'spec' in result
        rows.append({'dataset': result['label'], 'config': str(result['config']),
                     'images': result['images']._asdict(), 'ground_truth': result['gt']._asdict() if spec else None,
                     'models': [{'model': item.model.label, 'status': item.status, 'location': item.location,
                                 'size': item.size, 'detail': item.detail}
                                for item in ([*result['sam'].values(), *result['models']] if spec else [])],
                     'partitions': {key: value._asdict() for key, value in result['partitions'].items()} if spec else None,
                     'can_partition': result['can_partition'], 'can_explain': result['can_explain']})
    return {'datasets': rows, 'packages': packages}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', '--configs', nargs='+', default=DEFAULT_CONFIGS)
    parser.add_argument('--verify-hash', action='store_true', help='Read pinned checkpoints and compare SHA-256; slower')
    parser.add_argument('--strict', action='store_true', help='Exit 1 unless every dataset is READY for XAI')
    parser.add_argument('--json', type=Path, help='Also write a JSON report')
    args = parser.parse_args(argv)
    results, packages, extension = build_report(args.config, args.verify_hash)
    print_report(results, packages, extension)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report_json(results, packages), indent=2) + '\n')
    blocked = any('BLOCKED' in [result['can_partition'], result['can_explain']] for result in results)
    if blocked or (args.strict and any(result['can_explain'] != 'READY' for result in results)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
