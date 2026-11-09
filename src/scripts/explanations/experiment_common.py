"""Shared dataset, explanation, evaluation, and reporting workflow for model runners."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
for directory in [ROOT, ROOT / 'src/scripts']:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
from config_paths import load_path_config, output_path, partition_path, resolve_path, evaluation_path
from model_adapters import load_adapter
from auc_export import auc_columns, enrich_auc_rows

METHODS = ['BPT', 'SAM_c', 'SAM_c_kernelshap']
DEFAULT_CONFIGS = {'OC_VGG': 'ImageNet_epito', 'OC_PartImageNet': 'PartImageNet_epito', 'OC_BUSI': 'BUSI_epito', 'OD_DETR': 'MSCOCO_epito', 'OC_ResNet': 'ImageNet_epito',
                   'OC_ViT': 'ImageNet_epito', 'COD_ViT': 'CAMO_epito', 'OC_CelebA': 'CelebAHQ_epito',
                   'OC_FunnyBirds': 'FunnyBirds_epito', 'OC_PACO': 'PACO_epito',
                   'OC_KvasirSEG': 'KvasirSEG_epito', 'OC_ISIC2018': 'ISIC2018_epito', 'OD_DOTA': 'DOTA_epito',
                   'VL_RefCOCO': 'RefCOCO_epito'}
MODEL_NAMES = {'OC_VGG': 'vgg16', 'OC_PartImageNet': 'resnet50', 'OC_BUSI': 'ghostsas001/busi-breast-ultrasound-siglip', 'OD_DETR': 'facebook/detr-resnet-50', 'OC_ResNet': 'resnet50',
               'OC_ViT': 'vit_b_16', 'COD_ViT': 'vit_b_16', 'OC_CelebA': 'celeba_cnn',
               'OC_FunnyBirds': 'resnet50', 'OC_PACO': 'resnet50',
               'OC_KvasirSEG': 'mmuratarat/kvasir-v2-classifier',
               'OC_ISIC2018': 'adeIaide/skin-lesion-7class-efficientnet', 'OD_DOTA': 'yolo11s-obb',
               'VL_RefCOCO': 'openai/clip-vit-base-patch32'}
# Runners that are not ImageNet classifiers, by the data.name their config must declare.
EXPECTED_DATASETS = {'OD_DETR': 'mscoco', 'COD_ViT': 'camo', 'OC_CelebA': 'celebahq', 'OC_BUSI': 'busi',
                     'OC_PartImageNet': 'partimagenet', 'OC_FunnyBirds': 'funnybirds', 'OC_PACO': 'paco',
                     'OC_KvasirSEG': 'kvasirseg', 'OC_ISIC2018': 'isic2018', 'OD_DOTA': 'dota',
                     'VL_RefCOCO': 'refcoco'}
# These load a local checkpoint from --checkpoint or the config's model.checkpoint.
CHECKPOINT_HELP = {'OC_CelebA': 'download_celeba_checkpoint.py',
                   **{kind: f'download_datasets.py --dataset {name} --assets model' for kind, name in
                      [('OC_FunnyBirds', 'funnybirds'), ('OC_KvasirSEG', 'kvasirseg'), ('OC_ISIC2018', 'isic2018'),
                       ('OD_DOTA', 'dota'), ('VL_RefCOCO', 'refcoco')]}}
LESION_RUNNERS = {'OC_BUSI', 'OC_KvasirSEG', 'OC_ISIC2018'}
# Detectors: a class score sums its detections, any detector weights are accepted, and GT follows the explained class.
DETECTOR_RUNNERS = {'OD_DETR', 'OD_DOTA'}
# Pointing accuracy and best IoU on every run with --show-gt, not only with --explain_all.
LOCALIZATION_RUNNERS = LESION_RUNNERS | {'OD_DOTA', 'VL_RefCOCO'}
# Vision-language: one record per referring expression; the adapter scores the image against it.
EXPRESSION_RUNNERS = {'VL_RefCOCO'}


def create_parser(kind, auc_background=True, road=True):
    parser = argparse.ArgumentParser(description=f'{kind}: BPT, BPT+SAM_c, SAM_c+Kernel SHAP')
    parser.add_argument('--config', default=DEFAULT_CONFIGS[kind])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--image-ids', nargs='+')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--model', default=MODEL_NAMES[kind])
    parser.add_argument('--checkpoint', type=Path, help='Classifier state_dict or local HF DETR directory')
    parser.add_argument('--weights', default='DEFAULT', help='Torchvision weights enum name, or none')
    parser.add_argument('--class-names', type=Path, help='JSON class names for a custom classifier')
    parser.add_argument('--local-files-only', action='store_true', help='Load HF DETR from cache only')
    parser.add_argument('--max-evals', '--budget', type=int, default=1000,
                         help='Single budget for the three partition-based methods unless explicitly overridden')
    parser.add_argument('--kernel-shap-samples', type=int)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--kernel-shap-batch-size', type=int)
    parser.add_argument('--model-batch-size', '--resnet-batch-size', '--vit-batch-size', '--detr-batch-size',
                        type=int, default=8, help='Maximum images in one model forward pass')
    parser.add_argument('--auc-batch-size', type=int, default=8)
    parser.add_argument('--auc_num_samples', '--auc-num-samples', type=int, default=200,
                         help='Requested points per insertion/deletion curve; repeated masks are deduplicated')
    parser.add_argument('--bg-type', choices=['black', 'gray', 'white', 'blurred', 'noise', 'full'], 
                        default='full', help='Background replacement for masked pixels (default: full)')
    if auc_background:
        parser.add_argument('--auc-bg-type', choices=['black', 'gray', 'white', 'blurred', 'noise', 'full'],
                            help='Background for AUC and other faithfulness metrics; default: same as --bg-type')
    parser.add_argument('--class-aggregation', choices=['sum', 'max'], default='sum')
    if kind in EXPRESSION_RUNNERS:
        parser.add_argument('--all-sentences', action='store_true',
                            help='Explain every expression of each referred object; default: the first one')
    if kind == 'OD_DOTA':
        parser.add_argument('--conf', type=float, default=.25,
                            help='YOLO-OBB confidence threshold (Ultralytics and OD_yolo default: 0.25)')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='DETR scoring threshold (default: 0 for continuous masked scores)')
    parser.add_argument('--detr-shortest-edge', type=int, help='Override the DETR processor resize')
    parser.add_argument('--attribute', help='CelebA attribute name; default Brown_Hair for OC_CelebA')
    parser.add_argument('--attribute-file', type=Path, help='CelebA-HQ-attribute.txt to read output label order')
    parser.add_argument('--target-class', type=int, help='Explain this output index instead of the top prediction')
    parser.add_argument('--show_gt', '--show-gt', action='store_true',
                        help='Show/save configured dataset segmentation GT')
    parser.add_argument('--gt-only', action='store_true',
                        help='Select only ImageNet images with an ImageNetS annotation, before --limit')
    parser.add_argument('--gt-class', help='Optional GT class: face part, ImageNetS synset/label ID, or foreground')
    parser.add_argument('--output-dir', type=Path, help='Override explanation base before bg_*_B*')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--no-auc', dest='compute_auc', action='store_false')
    if road:
        import road as road_metric
        parser.add_argument('--no-road', dest='compute_road', action='store_false',
                            help='Skip the ROAD scores (road_* columns); add them later with evaluate_road.py')
        road_metric.add_arguments(parser)
    parser.add_argument('--no-plots', dest='save_plots', action='store_false')
    parser.add_argument('--no-heatmaps', dest='save_heatmaps', action='store_false',
                        help='Skip individual heatmap PNGs; keep combined XAI plots and numeric attributions')
    parser.add_argument('--no-html', dest='build_html', action='store_false')
    parser.add_argument('--verbose-level', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--torch-threads', type=int, help='Optional CPU intra-op thread limit')
    from all_methods import add_arguments
    add_arguments(parser)
    return parser


def comparison_names(args):
    from all_methods import method_names
    return method_names(args)


def normalize_id(value, data):
    value = Path(str(value)).stem
    if data.get('input_format') != 'directory' and value.isdigit():
        return f'{int(value):012d}'
    if data.get('name', '').lower() == 'imagenet' and value.isdigit():
        return f'ILSVRC2012_val_{int(value):08d}'
    return value


def prepare_inputs(kind, args):
    from all_methods import apply_budget_selection
    apply_budget_selection(args, kind)
    if args.explain_all:
        from all_methods import validate
        validate(args)
    config = load_path_config(args.config)
    data = config['data']
    expected = EXPECTED_DATASETS.get(kind, 'imagenet')
    if data.get('name', '').replace('_', '').lower() != expected:
        raise ValueError(f'{kind} requires the {expected} dataset config')
    if args.gt_only and (not args.show_gt or expected != 'imagenet'):
        raise ValueError('--gt-only requires an ImageNet runner with --show-gt')
    if expected in {'partimagenet', 'paco'} and args.show_gt:
        gt_annotation = data.get('ground_truth', {}).get('annotation_file')
        if not gt_annotation or resolve_path(gt_annotation) != resolve_path(data['annotation_file']):
            raise ValueError(f"{data['name']} data.annotation_file and ground_truth.annotation_file must select the same split")
    if args.show_gt and kind != 'OD_DETR':
        from ground_truth import validate_config
        validate_config(config)
    for name in ['max_evals', 'eval_batch_size', 'model_batch_size', 'auc_batch_size']:
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if args.auc_num_samples < 2:
        raise ValueError('--auc-num-samples must be at least 2')
    for name in ['kernel_shap_samples', 'kernel_shap_batch_size', 'torch_threads', 'detr_shortest_edge']:
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    folder = partition_path(config)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    image_dir = resolve_path(data['image_dir'])
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    annotation_file = None
    if expected == 'mscoco':
        annotation_file = resolve_path(data['annotation_file'])
        items = json.loads(annotation_file.read_text())['images']
        images = {f"{int(item['id']):012d}": image_dir / item['file_name'] for item in items}
    elif expected in {'partimagenet', 'paco'}:
        from partimagenet_data import collect_images
        images = {item['id']: image_dir / item['file_name'] for item in collect_images(data)}
    elif expected == 'busi':
        from src.medical.datasets import discover_images
        images = {item['id']: image_dir / item['file_name'] for item in discover_images(image_dir)}
    else:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        images = {}
        for path in sorted(image_dir.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if path.stem in images:
                raise ValueError(f'Duplicate input image stem: {path.stem}')
            images[path.stem] = path
    completed = {}
    for path in sorted(folder.glob('*_results.json')):
        image_id = path.name.removesuffix('_results.json')
        if image_id not in images:
            continue
        try:
            metadata = json.loads(path.read_text())
            saved = metadata.get('saved_partitions')
            stages = list(saved or {}) or metadata.get('generation_settings', {}).get('save_partitions') or ['sam', 'refined']
            if saved and 'compact' not in saved:
                continue
            if (folder / f'{image_id}_compact.npy').is_file() and all(
                    (folder / f'{image_id}_{stage}.npy').is_file() for stage in stages):
                completed[image_id] = metadata
        except (ValueError, TypeError, AttributeError):
            continue
    ids = [normalize_id(value, data) for value in args.image_ids] if args.image_ids else sorted(completed)
    ids = list(dict.fromkeys(ids))
    if args.show_gt and expected == 'imagenet':
        from ground_truth import validate_config, _mask_index
        gt_settings, gt_root = validate_config(config)
        variant = str(gt_settings.get('variant', 919))
        available = _mask_index(str(gt_root), 'imagenets', variant)
        matched = [image_id for image_id in ids if image_id in available]
        print(f'ImageNetS{variant} GT: {len(available)} indexed masks; '
              f'{len(matched)}/{len(ids)} selected inputs have annotations; root: {gt_root}')
        if not available:
            raise ValueError(f'No ImageNetS{variant} annotation masks found below {gt_root}; '
                             'check ground_truth.root and variant')
        if args.gt_only:
            ids = matched
            if not ids:
                raise ValueError('No completed partitions overlap the ImageNetS annotations; '
                                 'generate partitions for annotated images or select the correct GT variant')
        elif len(matched) < len(ids):
            print('Some images have no ImageNetS annotation. Use --gt-only to select annotated images before --limit.')
    if expected == 'refcoco':
        records = expression_records(data, completed, images, folder, args)
    else:
        if args.limit is not None:
            ids = ids[:args.limit]
        if not ids:
            raise ValueError(f'No completed SAM_c inputs in {folder}; generate --save-partitions compact first')
        records = []
        for image_id in ids:
            if image_id not in completed:
                raise ValueError(f'{image_id}: missing image, completed metadata, or compact partition')
            if not images[image_id].is_file():
                raise FileNotFoundError(images[image_id])
            records.append({'image_id': image_id, 'image_path': images[image_id],
                            'partition_path': folder / f'{image_id}_compact.npy',
                            'partition_metadata': completed[image_id]})
    if args.attribute and kind != 'OC_CelebA':
        raise ValueError('--attribute is for OC_CelebA')
    if kind in CHECKPOINT_HELP and not (kind in DETECTOR_RUNNERS and args.model != MODEL_NAMES[kind]):
        # Other detector weights (e.g. --model yolo11m-obb) are looked up by name, not the config's file.
        args.checkpoint = args.checkpoint or config.get('model', {}).get('checkpoint')
        if not args.checkpoint:
            raise ValueError(f'Set --checkpoint or model.checkpoint; run {CHECKPOINT_HELP[kind]} first')
    if kind == 'OC_CelebA':
        if args.attribute and args.target_class is not None:
            raise ValueError('Choose --attribute or --target-class, not both')
    for name in ['checkpoint', 'class_names', 'attribute_file']:
        if getattr(args, name):
            setattr(args, name, resolve_path(getattr(args, name)))
            if not getattr(args, name).exists():
                raise FileNotFoundError(getattr(args, name))
    if kind not in DETECTOR_RUNNERS and args.model != MODEL_NAMES[kind]:
        raise ValueError(f'{kind} supports --model {MODEL_NAMES[kind]}')
    base = resolve_path(args.output_dir) if args.output_dir else output_path(config, 'explanations') / kind / folder.name
    if kind == 'OC_CelebA':
        attribute_folder = args.attribute or ('Brown_Hair' if args.target_class is None else f'class_{args.target_class}')
        if not attribute_folder.replace('_', '').isalnum():
            raise ValueError('Attribute names must contain letters, numbers, and underscores')
        base = base / attribute_folder
    run_dir = base / run_folder(args)
    return {'config': config, 'records': records, 'partition_dir': folder,
            'run_dir': run_dir, 'evaluation_dir': evaluation_path(config, run_dir),
            'annotation_file': annotation_file, 'methods': comparison_names(args) if args.explain_all else list(METHODS),
            'ground_truth': ({'format': 'coco', 'annotation_file': annotation_file}
                             if expected == 'mscoco' else data.get('ground_truth')) if args.show_gt else None}


def expression_records(data, completed, images, folder, args):
    """One record per referring expression; records of one image share its partition.

    Ordered so that --limit N first takes one object from each of N images.
    """
    from refcoco_data import load_expressions, select_expressions
    entries = select_expressions(load_expressions(resolve_path(data['expressions'])), args.all_sentences)
    if args.image_ids:
        wanted = set(args.image_ids)
        entries = [entry for entry in entries if entry['record_id'] in wanted or entry['image_stem'] in wanted]
        unknown = wanted - {entry['record_id'] for entry in entries} - {entry['image_stem'] for entry in entries}
        if unknown:
            raise ValueError(f'Unknown RefCOCO record or image IDs: {sorted(unknown)[:5]}')
        missing = sorted({entry['image_stem'] for entry in entries} - set(completed))
        if missing:
            raise ValueError(f'{missing[0]}: missing image, completed metadata, or compact partition')
    rank, seen = {}, {}
    for entry in entries:
        key = (entry['image_stem'], entry['ref_id'])
        if key not in rank:
            rank[key] = seen.get(entry['image_stem'], 0)
            seen[entry['image_stem']] = rank[key] + 1
    entries = sorted((entry for entry in entries if entry['image_stem'] in completed),
                     key=lambda entry: (entry['sentence_index'], rank[(entry['image_stem'], entry['ref_id'])],
                                        entry['image_stem'], entry['ref_id']))
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries:
        raise ValueError(f'No completed SAM_c inputs in {folder}; generate --save-partitions compact first')
    records = []
    for entry in entries:
        stem = entry['image_stem']
        if not images[stem].is_file():
            raise FileNotFoundError(images[stem])
        records.append({'image_id': entry['record_id'], 'partition_id': stem, 'image_path': images[stem],
                        'partition_path': folder / f'{stem}_compact.npy', 'partition_metadata': completed[stem],
                        **{key: entry[key] for key in ['expression', 'ann_id', 'ref_id', 'sent_id', 'category']}})
    return records


def auc_background(args):
    """The AUC background when it differs from the explanation's, otherwise None."""
    value = getattr(args, 'auc_bg_type', None)
    return None if value == args.bg_type else value


def run_folder(args):
    auc = auc_background(args)
    return f"bg_{args.bg_type}{f'_auc_{auc}' if auc else ''}_B{args.max_evals}"


def auc_scoring(adapter, image, args, target, predict, f_s, f_0):
    """Predictor and baseline scores for evaluation; f_0 depends on the background used."""
    mode = auc_background(args)
    if mode is None:
        return predict, f_s, f_0
    scoring = make_masked_predictor(adapter, image, backgrounds(image, mode), args.model_batch_size)
    keep_all = np.ones(image.shape[:2], dtype=bool)
    scores = scoring(np.stack([keep_all, ~keep_all]))[:, target]
    return scoring, float(scores[0]), float(scores[1])


def backgrounds(image, mode):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    def make(name):
        if name in {'black', 'gray', 'white'}:
            return np.full_like(image, {'black': 0, 'gray': 127, 'white': 255}[name])
        if name == 'blurred':
            return gaussian_filter(image.astype(float), sigma=(8, 8, 0)).astype(np.uint8)
        if name == 'noise':
            return np.clip(gaussian_filter(rng.normal(128, 128, image.shape), sigma=(2, 2, 0)), 0, 255).astype(np.uint8)
        raise ValueError(f'Unknown background: {name}')
    return [make(name) for name in ['black', 'gray', 'white', 'blurred', 'noise']] if mode == 'full' else [make(mode)]


def make_masked_predictor(adapter, image, replacements, batch_size):
    def predict(masks):
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 3 or masks.shape[1:] != image.shape[:2]:
            raise ValueError(f'Expected [N,{image.shape[0]},{image.shape[1]}] masks, got {masks.shape}')
        predictions = []
        step = max(1, batch_size // len(replacements))
        for start in range(0, len(masks), step):
            chunk = masks[start:start + step]
            images = [np.where(mask[..., None], image, background)
                      for mask in chunk for background in replacements]
            scores = np.asarray(adapter.predict_batch(images), dtype=float)
            if scores.shape != (len(images), len(adapter.class_names)) or not np.isfinite(scores).all():
                raise ValueError('Model returned invalid class scores')
            predictions.append(scores.reshape(len(chunk), len(replacements), -1).mean(axis=1))
        return np.concatenate(predictions) if predictions else np.empty((0, len(adapter.class_names)))
    return predict


def load_runtime():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import shap_bpt
    from tqdm import tqdm
    # Legacy utility imports locate the project from cwd. Scope that requirement
    # to the import rather than changing the caller's working directory.
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        import utils_xai as utx
        import utils_sam_2 as partitions
    finally:
        os.chdir(previous)
    return SimpleRuntime(plt, pd, torch, shap_bpt, tqdm, utx, partitions)


class SimpleRuntime:
    def __init__(self, plt, pd, torch, shap_bpt, tqdm, utx, partitions):
        self.plt, self.pd, self.torch, self.shap_bpt = plt, pd, torch, shap_bpt
        self.tqdm, self.utx, self.partitions = tqdm, utx, partitions


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2) + '\n')
    temporary.replace(path)


def run_signature(kind, args, config=None):
    # ROAD only reads the finished maps, so its options never invalidate finished explanations.
    ignored = {'limit', 'image_ids', 'dry_run', 'resume', 'verbose_level', 'fail_fast', 'torch_threads', 'output_dir',
               'compute_road', 'road_fractions', 'road_noise', 'road_seed', 'road_short_side'}
    settings = {key: jsonable(value) for key, value in vars(args).items() if key not in ignored}
    if not args.explain_all:
        for option in ['explain_all', 'bpt_budgets', 'lime_budgets', 'gradient_samples',
                       'idg_steps', 'attribution_batch_size', 'explanation_seed', 'aa_budget', 'lime_segments']:
            settings.pop(option, None)
    if settings.get('save_heatmaps', True):
        settings.pop('save_heatmaps', None)  # Preserve signatures from before this option existed.
    for option in ['attribute', 'attribute_file', 'budget_selection']:
        if settings.get(option) is None:
            settings.pop(option, None)
    if auc_background(args) is None:
        settings.pop('auc_bg_type', None)  # Preserve signatures from before this option existed.
    if settings.get('gt_class') is None:
        settings.pop('gt_class', None)
    if args.show_gt and config and config['data'].get('ground_truth'):
        settings['ground_truth'] = config['data']['ground_truth']
    for name in ['checkpoint', 'class_names', 'attribute_file']:
        path = getattr(args, name)
        if path and path.is_file():
            stat = path.stat()
            settings[name + '_stamp'] = [stat.st_size, stat.st_mtime_ns]
    return hashlib.sha256(json.dumps({'kind': kind, **settings}, sort_keys=True).encode()).hexdigest()


def input_stamp(record):
    return {key: [str(record[key]), record[key].stat().st_size, record[key].stat().st_mtime_ns]
            for key in ['image_path', 'partition_path']}


def completed_manifest(folder, image_id, signature, stamp):
    path = folder / f'{image_id}_complete.json'
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text())
        if metadata.get('signature') != signature or metadata.get('input_stamp') != stamp:
            raise ValueError(f'{image_id}: run settings/inputs changed; use --no-resume or another --output-dir')
        if all((folder / name).is_file() for name in metadata['artifacts']):
            return metadata
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return None


def evaluate(runtime, method, values, predict, target, f_s, f_0, args, explanation_seconds):
    start = time.perf_counter()
    heatmap = np.asarray(values[0], dtype=float)
    result = {'label': method, 'time_exp': explanation_seconds}
    for direction in ['del', 'ins']:
        result['auc_' + direction] = runtime.utx.saliency_to_auc(
            predict, heatmap, f_s, f_0, target, batch_size=args.auc_batch_size,
            method=direction, num_samples=args.auc_num_samples)
    result.update(auc_requested_points=args.auc_num_samples,
                  auc_ins_points=len(result['auc_ins']['xs']), auc_del_points=len(result['auc_del']['xs']))
    if args.verbose_level != 'low':
        print(f"{method} AUC points: requested={args.auc_num_samples}; "
              f"insertion={result['auc_ins_points']}, deletion={result['auc_del_points']}", flush=True)
    result.update(runtime.utx.compute_extra_faithfulness_metrics(
        predict, heatmap, f_s, f_0, target, batch_size=args.auc_batch_size, sensitivity_seed=0))
    result['time_eval'] = time.perf_counter() - start
    return result


def bpt_partition_labels(tree, region_count, shape):
    """Cut the actual explanation tree at the SAM region count for display."""
    import heapq
    nodes = [-(int(tree.N) - 1)]
    while len(nodes) < min(region_count, int(tree.U)):
        node = -heapq.heappop(nodes)
        if node < tree.U:
            heapq.heappush(nodes, -node)
            break
        offset = node - tree.U
        heapq.heappush(nodes, -int(tree.cl_left[offset]))
        heapq.heappush(nodes, -int(tree.cl_right[offset]))
    labels = np.empty(shape, dtype=np.int32)
    for label, negative in enumerate(sorted(nodes)):
        node = -negative
        if node < tree.U:
            start, end = int(tree.leaf_idx[node]), int(tree.leaf_idx[node]) + 1
        else:
            start, end = int(tree.cl_start[node-tree.U]), int(tree.cl_end[node-tree.U])
        labels.ravel()[np.asarray(tree.pixels[start:end], dtype=int)] = label
    return labels


def explanation_grid(runtime, image, partition, attributions, prediction, output,
                     ground_truth=None, bpt_partition=None, auc_results=None, display_methods=None):
    plt = runtime.plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    try:
        axes[0, 0].imshow(image)
        for detection in prediction.get('detections', []):
            if detection['class_id'] != prediction['predicted_class_id']:
                continue
            from matplotlib.patches import Polygon, Rectangle
            if detection.get('polygon'):  # Oriented boxes (OD_DOTA)
                axes[0, 0].add_patch(Polygon(detection['polygon'], closed=True, fill=False, edgecolor='lime', linewidth=1))
                continue
            x1, y1, x2, y2 = detection['box']
            axes[0, 0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=1))
        axes[0, 0].set_title('Input')
        for ax, labels, name in [(axes[0, 1], bpt_partition, 'BPT'), (axes[0, 2], partition, 'SAM_c')]:
            if labels is not None:
                ax.imshow(labels, cmap='turbo', interpolation='nearest')
                ax.set_title(f'{name}: {len(np.unique(labels))} regions')
        for ax, method in zip(axes[1], display_methods or METHODS):
            if method not in attributions:
                continue
            heatmap = np.asarray(attributions[method][0])
            from attribution_visuals import notebook_style, draw_gt_boundaries
            handle = ax.imshow(heatmap, origin='upper', **notebook_style(heatmap))
            ax.set_title(f'Attributions-{method}')
            if ground_truth is not None:
                mask = np.asarray(ground_truth['mask'], dtype=bool)
                if mask.shape != heatmap.shape:
                    raise ValueError('GT mask and attribution dimensions must match')
                if mask.any():
                    draw_gt_boundaries(ax, ground_truth)
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            colorbar_ax = divider.append_axes('right', size='4%', pad=.08)
            fig.colorbar(handle, cax=colorbar_ax)
        for ax in axes[:2].ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        top_classes = prediction.get('top_k_classes') or []
        top = max(top_classes, key=lambda item: float(item['confidence'])) if top_classes else None
        predicted_title = (f"Top predicted: {top['class_name']} ({float(top['confidence']):.4f})"
                           if top else 'Top predicted: unavailable')
        from ground_truth import semantic_styles, draw_semantic_legend
        if ground_truth is not None:
            draw_semantic_legend(axes[0, 0], ground_truth, inside=True)
        gt_note = ((' | GT boundaries: semantic colors' if semantic_styles(ground_truth) else ' | GT boundary: black') if ground_truth is not None and np.any(ground_truth['mask'])
                   else ' | GT: ' + ground_truth['status'] if ground_truth is not None else '')
        fig.suptitle(f"{predicted_title} | Explained: {prediction['explained_class']}{gt_note}")
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(output, dpi=130)
    finally:
        plt.close(fig)


def build_report(runtime, run_dir, evaluation_dir, rows, summaries, save_plots, build_html):
    pd = runtime.pd
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv = evaluation_dir / 'auc_results_all_images_current_run.csv'
    frame.to_csv(csv, index=False)
    frame.to_csv(run_dir / csv.name, index=False)
    metrics = [column for column in ['auc_ins', 'auc_del', 'road_gap', 'road_morf', 'road_lerf', 'time_exp',
                                    'time_eval', 'sensitivity_n_corr', 'stability_top10_jaccard', 'best_iou',
                                    'pointing_accuracy'] if column in frame]
    if not frame.empty:
        frame.groupby('method', sort=False)[metrics].agg(['mean', 'std', 'median']).to_csv(
            evaluation_dir / 'method_summary.csv')
        # Who wins: leaderboard.csv, metric_winners.csv, winner.md, method_boxplots.png, method_ranks.png.
        try:
            import xai_leaderboard
            title = ' / '.join(Path(evaluation_dir).parts[-3:])
            result = xai_leaderboard.dataset_report(xai_leaderboard.use_relative_road(frame), evaluation_dir,
                                                    title, save_plots)
            if result is not None:
                print((evaluation_dir / 'winner.md').read_text(), flush=True)
        except Exception:
            print('Leaderboard failed; the results are saved. Rebuild it with xai_leaderboard.py dataset:\n'
                  + traceback.format_exc(), file=sys.stderr, flush=True)
    if not build_html:
        return
    sections = []
    for summary in summaries:
        folder = run_dir / summary['image_id']
        completion = json.loads((folder / f"{summary['image_id']}_complete.json").read_text())
        permitted = set(completion['artifacts'])
        images = []
        for path in sorted(folder.glob('*.png')):
            if path.name not in permitted:
                continue
            relative = os.path.relpath(path, evaluation_dir).replace(os.sep, '/')
            images.append(f'<figure><figcaption>{html.escape(path.stem)}</figcaption>'
                          f'<img loading="lazy" src="{html.escape(relative)}"></figure>')
        sections.append(f"<section><h2>{html.escape(summary['image_id'])}: "
                        f"{html.escape(summary['explained_class'])}</h2>{''.join(images)}</section>")
    table = frame.to_html(index=False, escape=True) if not frame.empty else '<p>No completed images.</p>'
    document = ('<!doctype html><meta charset="utf-8"><title>XAI results</title>'
                '<style>body{font:14px sans-serif;margin:24px}img{max-width:100%}'
                'figure{overflow-x:auto}img[src$="_all_methods.png"]{max-width:none}'
                'table{border-collapse:collapse}td,th{padding:6px}section{margin:24px 0}</style>'
                f'<h1>XAI results</h1><p>{len(summaries)} completed images</p>{table}{"".join(sections)}')
    (evaluation_dir / 'xai_results_report.html').write_text(document)


def run(kind, args, adapter=None):
    run_started = time.perf_counter()
    inputs = prepare_inputs(kind, args)
    print(json.dumps({key: jsonable(value) for key, value in inputs.items() if key not in {'records', 'config'}}, indent=2))
    print('Images:', ', '.join(record['image_id'] for record in inputs['records']))
    if args.dry_run:
        return inputs
    print(f'AUC: {args.auc_num_samples} requested points per curve' if args.compute_auc else 'AUC: disabled')
    road_config = None
    if getattr(args, 'compute_road', False):
        import road as road_metric
        road_config = road_metric.settings(args)
        print(f"ROAD: {len(road_config['fractions'])} removal levels, short side {road_config['short_side'] or 'full'}")
    else:
        print('ROAD: disabled')
    runtime = load_runtime()
    if args.torch_threads:
        runtime.torch.set_num_threads(args.torch_threads)
    device_name = args.device or inputs['config'].get('device', 'auto')
    if device_name == 'auto':
        device_name = ('cuda' if runtime.torch.cuda.is_available() else
                       'mps' if runtime.torch.backends.mps.is_available() else 'cpu')
    device = runtime.torch.device(device_name)
    run_dir, evaluations = inputs['run_dir'], inputs['evaluation_dir']
    signature = run_signature(kind, args, inputs['config'])
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(parents=True, exist_ok=True)
    prior, pending = [], []
    for record in inputs['records']:
        folder = run_dir / record['image_id']
        manifest = completed_manifest(folder, record['image_id'], signature, input_stamp(record)) if args.resume else None
        if manifest is not None:
            prior.append(manifest)
        else:
            pending.append(record)
    print(f'Pending: {len(pending)}; complete: {len(prior)}; device: {device}')
    model_load_sec = 0.0
    if pending and adapter is None:
        model_started = time.perf_counter()
        adapter = load_adapter(kind, args, device)
        model_load_sec = time.perf_counter() - model_started
    write_json(run_dir / 'run_info.json', {'runner': kind, 'model': args.model, 'signature': signature,
               'methods': inputs['methods'], 'config': args.config, 'masks_base_path': str(inputs['partition_dir']),
               'image_ids': [record['image_id'] for record in inputs['records']],
               'background_replacement_values': args.bg_type, 'auc_background': auc_background(args) or args.bg_type,
               'max_evals': args.max_evals,
               'results_dir': run_dir, 'evaluations_dir': evaluations, 'settings': vars(args),
               'ground_truth': inputs['ground_truth'],
               'score_type': 'detection confidence' if kind in DETECTOR_RUNNERS else
               'image-text cosine similarity' if kind in EXPRESSION_RUNNERS else 'attribute logit' if kind == 'OC_CelebA' else 'classification probability'})
    rows = [row for manifest in prior for row in manifest['rows']]
    for manifest in prior:
        image_id = manifest['prediction']['image_id']
        metrics = run_dir / image_id / f'{image_id}_metrics.json'
        if metrics.is_file():
            enrich_auc_rows(manifest['rows'], json.loads(metrics.read_text()))
    if road_config is not None:
        from evaluate_road import saved_columns
        missing = []
        for manifest in prior:
            image_id = manifest['prediction']['image_id']
            columns = saved_columns(run_dir / image_id / f'{image_id}_road.json', road_config)
            for row in manifest['rows']:
                row.update(columns.get(row['method'], {}))
            if any(row['method'] not in columns for row in manifest['rows']):
                missing.append(image_id)
        if missing:
            print(f'{len(missing)} finished images have no ROAD scores yet (first: {missing[0]}). '
                  f'Add them with evaluate_road.py and the same options.', flush=True)
    summaries = [manifest['prediction'] for manifest in prior]
    failures = []
    skipped = []
    image_timings = []
    from PIL import Image
    for record in runtime.tqdm(pending, desc=kind, ascii=True, dynamic_ncols=True):
        image_id = record['image_id']
        folder = run_dir / image_id
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / f'{image_id}_complete.json'
        if marker.exists():
            marker.unlink()
        try:
            started = time.perf_counter()
            with Image.open(record['image_path']) as source:
                image = np.asarray(source.convert('RGB'))
            _, labels = runtime.partitions.load_refine_partitions(
                image, str(inputs['partition_dir']), record.get('partition_id', image_id), 'compact')
            if hasattr(adapter, 'set_record'):
                adapter.set_record(record)  # The referring expression this record explains.
            replacements = backgrounds(image, args.bg_type)
            predict = make_masked_predictor(adapter, image, replacements, args.model_batch_size)
            full = predict(np.ones((1, *image.shape[:2]), dtype=bool))[0]
            if kind == 'OC_CelebA' and args.target_class is None:
                attribute = args.attribute or 'Brown_Hair'
                if attribute not in adapter.class_names:
                    raise ValueError(f'Unknown CelebA attribute: {attribute}')
                target = adapter.class_names.index(attribute)
            else:
                if kind == 'OD_DOTA' and args.target_class is None and not np.any(full > 0):
                    # No box above --conf: there is no detected class to explain. Not lowered automatically.
                    reason = f'no detections above conf {args.conf}'
                    write_json(folder / f'{image_id}_skipped.json', {'image_id': image_id, 'reason': reason})
                    skipped.append({'image_id': image_id, 'reason': reason})
                    print(f'{image_id}: skipped, {reason}', flush=True)
                    continue
                target = int(np.argmax(full)) if args.target_class is None else args.target_class
            if target < 0 or target >= len(adapter.class_names):
                raise ValueError(f'Target class {target} outside model output range')
            explainer = runtime.shap_bpt.Explainer(predict, image, num_explained_classes=None,
                                                  explained_class_ids=[target], verbose=args.verbose_level == 'high')
            f_s, f_0 = float(explainer.base_nuN[0]), float(explainer.base_nu0[0])
            auc_predict, auc_f_s, auc_f_0 = auc_scoring(adapter, image, args, target, predict, f_s, f_0)
            prediction = {'image_id': image_id, 'source_image': str(record['image_path']),
                          'fixed_category': adapter.class_names[target], 'explained_class': adapter.class_names[target],
                          'predicted_class_id': target, 'f_S': f_s, 'f_0': f_0,
                          'top_k_classes': [{'class_id': int(index), 'class_name': adapter.class_names[int(index)],
                                             'confidence': float(full[index])} for index in np.argsort(full)[::-1][:5]],
                          'detections': adapter.detections(image)}
            if kind in EXPRESSION_RUNNERS:
                prediction.update({key: record[key] for key in ['expression', 'ann_id', 'ref_id', 'sent_id', 'category']})
                prediction['image_stem'] = record['partition_id']
            ground_truth = None
            if args.show_gt:
                from ground_truth import load_ground_truth, save_ground_truth
                gt_class = ((args.gt_class or prediction['explained_class']) if kind in DETECTOR_RUNNERS else
                            str(record['ann_id']) if kind in EXPRESSION_RUNNERS and args.gt_class is None else args.gt_class)
                ground_truth = load_ground_truth(
                    inputs['config'], record['image_path'], image.shape, gt_class=gt_class,
                    predicted_index=target, label_space=getattr(adapter, 'label_space', None),
                    predicted_class=prediction['explained_class'])
                prediction['ground_truth'] = save_ground_truth(ground_truth, image, folder, image_id)
                if args.verbose_level != 'low':
                    print(f"{image_id} GT: {ground_truth['class_name']}; {ground_truth['status']}; "
                          f"foreground pixels={ground_truth['foreground_pixels']}", flush=True)
            write_json(folder / f'{image_id}_predictions.json', prediction)
            attributions, image_rows, auc_results = {}, [], []
            bpt_labels = None
            bpt_tree = None
            statuses = {}
            if args.explain_all:
                if args.verbose_level != 'low':
                    print(f'{image_id}: building BPT tree', flush=True)
                bpt_tree = runtime.shap_bpt.build_bpt_from_image(image)
                bpt_labels = bpt_partition_labels(bpt_tree, len(np.unique(labels)), image.shape[:2])
            for method in ([] if args.explain_all else METHODS):
                start = time.perf_counter()
                if method == 'SAM_c_kernelshap':
                    values = (np.full((1, *labels.shape), f_s - f_0) if len(np.unique(labels)) == 1 else
                              runtime.utx.kernel_shap_partition_regions(
                                  predict, labels, target, n_samples=args.kernel_shap_samples or args.max_evals,
                                  batch_size=args.kernel_shap_batch_size or args.eval_batch_size, seed=0))
                else:
                    tree = (runtime.shap_bpt.build_bpt_from_image(image, prebuilt_partitions=labels)
                            if method == 'SAM_c' else runtime.shap_bpt.build_bpt_from_image(image))
                    if method == 'BPT':
                        bpt_tree = tree
                        bpt_labels = bpt_partition_labels(tree, len(np.unique(labels)), image.shape[:2])
                    values = explainer.explain_instance(args.max_evals, bpt=tree, method='BPT', batch_size=args.eval_batch_size)
                elapsed = time.perf_counter() - start
                values = np.asarray(values, dtype=float)
                if values.shape != (1, *image.shape[:2]) or not np.isfinite(values).all():
                    raise ValueError(f'{method}: invalid attribution map {values.shape}')
                attributions[method] = values
                statuses[method] = {'status': 'complete'}
                row = {'image_id': image_id, 'image_no': image_id, 'method': method,
                       'fixed_category': prediction['explained_class'], 'f_S': f_s, 'f_0': f_0,
                       'time_exp': elapsed, 'time_eval': 0.0}
                if args.compute_auc:
                    result = evaluate(runtime, method, values, auc_predict, target, auc_f_s, auc_f_0, args, elapsed)
                    auc_results.append(result)
                    row.update(auc_columns(result))
                    row.update({key: value for key, value in result.items()
                                if isinstance(value, (int, float, np.number))})
                image_rows.append(row)
                if args.save_plots and args.save_heatmaps:
                    runtime.utx.plot_single_attributions(values, str(folder), f'{method}_{image_id}.png',
                                                        save_plot=True, destroy_fig=True, notebook_style=True)
            if args.explain_all:
                from all_methods import compute_extra
                for method, values, info, elapsed in compute_extra(
                        runtime, args, adapter, image, replacements, explainer, bpt_tree, target,
                        labels=labels, predict=predict, dataset_name=inputs['config']['data']['name']):
                    statuses[method] = info
                    if values is None:
                        print(f'{image_id} {method}: {info["status"]}: {info.get("reason", "")}')
                        continue
                    attributions[method] = values
                    row = {'image_id': image_id, 'image_no': image_id, 'method': method,
                           'fixed_category': prediction['explained_class'], 'f_S': f_s, 'f_0': f_0,
                           'time_exp': elapsed, 'time_eval': 0.0}
                    if args.compute_auc:
                        result = evaluate(runtime, method, values, auc_predict, target, auc_f_s, auc_f_0, args, elapsed)
                        auc_results.append(result)
                        row.update(auc_columns(result))
                        row.update({key: value for key, value in result.items()
                                    if isinstance(value, (int, float, np.number))})
                    image_rows.append(row)
                    if args.save_plots and args.save_heatmaps:
                        runtime.utx.plot_single_attributions(values, str(folder), f'{method}_{image_id}.png',
                                                            save_plot=True, destroy_fig=True, notebook_style=True)
                from all_methods import best_iou
                for method, values in attributions.items():
                    result = best_iou(values[0], ground_truth)
                    if result:
                        result.pop('mask')
                        statuses[method].update(result)
                        next(row for row in image_rows if row['method'] == method).update(result)
                write_json(folder / f'{image_id}_method_status.json', statuses)
            if kind in LOCALIZATION_RUNNERS and ground_truth is not None:
                from src.medical.evaluation import pointing_accuracy
                from all_methods import best_iou
                for method, values in attributions.items():
                    localization = {'pointing_accuracy': pointing_accuracy(values[0], ground_truth['mask'])}
                    diagnostic = best_iou(values[0], ground_truth)
                    if diagnostic:
                        diagnostic.pop('mask')
                        localization.update(diagnostic)
                    for row in [*image_rows, *auc_results]:
                        if row.get('method', row.get('label')) == method:
                            row.update(localization)
            if road_config is not None:
                from evaluate_road import road_columns, score_maps
                if args.verbose_level != 'low':
                    print(f'{image_id}: ROAD for {len(attributions)} maps', flush=True)
                road_path = folder / f'{image_id}_road.json'
                road_path.unlink(missing_ok=True)  # The maps are new, so earlier ROAD scores no longer apply.
                scores = score_maps(adapter.predict_batch, image, {name: values[0] for name, values in attributions.items()},
                                    target, image_id, road_config, road_path,
                                    verbose=args.verbose_level == 'high')
                for row in [*image_rows, *auc_results]:
                    method = row.get('method', row.get('label'))
                    if method in scores:
                        row.update(road_columns(scores[method]))
            np.savez_compressed(folder / f'{image_id}_attributions.npz', **attributions)
            write_json(folder / f'{image_id}_metrics.json', auc_results if args.compute_auc else image_rows)
            if args.save_plots:
                from all_methods import grid_methods
                explanation_grid(runtime, image, labels, attributions, prediction,
                                 folder / f'{image_id}_xai_results.png', ground_truth, bpt_labels, auc_results,
                                 display_methods=grid_methods(args) if args.explain_all else None)
                if args.explain_all:
                    from all_methods import comparison_plot
                    comparison_plot(runtime, image, attributions, prediction,
                                    folder / f'{image_id}_all_methods.png', ground_truth, statuses)
                if args.compute_auc:
                    runtime.utx.plot_auc_results(auc_results, str(folder), save_plot=True,
                                                 image_no=image_id, destroy_figs=True)
                    runtime.utx.plot_faithfulness_curves(auc_results, str(folder), save_plot=True,
                                                       image_no=image_id, destroy_figs=True)
            artifacts = [f'{image_id}_predictions.json', f'{image_id}_attributions.npz', f'{image_id}_metrics.json']
            if args.explain_all:
                artifacts += [f'{image_id}_method_status.json']
                if args.save_plots:
                    artifacts += [f'{image_id}_all_methods.png']
            if args.save_plots:
                artifacts += [f'{image_id}_xai_results.png']
                if args.save_heatmaps:
                    artifacts += [f'{method}_{image_id}.png' for method in attributions]
                if args.compute_auc:
                    artifacts += [f'auc_results_{image_id}.png', f'faithfulness_curves_{image_id}.png']
            if args.show_gt:
                artifacts += [f'{image_id}_gt{extension}' for extension in ['.npy', '.png', '.json', '_labels.npz']]
            image_timing = {'image_id': image_id, 'total_time_sec': time.perf_counter() - started,
                            'explanation_time_sec': sum(float(row['time_exp']) for row in image_rows),
                            'evaluation_time_sec': sum(float(row['time_eval']) for row in image_rows)}
            manifest = {'signature': signature, 'input_stamp': input_stamp(record), 'artifacts': artifacts,
                        'rows': image_rows, 'prediction': prediction, 'total_time_sec': image_timing['total_time_sec'],
                        'timing': image_timing}
            write_json(marker, manifest)  # Completion marker is always written last.
            image_timings.append(image_timing)
            rows.extend(image_rows)
            summaries.append(prediction)
            runtime.pd.DataFrame(rows).to_csv(run_dir / 'auc_results_all_images_current_run.csv', index=False)
            runtime.pd.DataFrame(rows).to_csv(evaluations / 'auc_results_all_images_current_run.csv', index=False)
        except Exception:
            failure = {'image_id': image_id, 'traceback': traceback.format_exc()}
            failures.append(failure)
            print(failure['traceback'], file=sys.stderr)
            if args.fail_fast:
                write_json(run_dir / 'failures.json', failures)
                raise
    write_json(run_dir / 'failures.json', failures)
    if kind == 'OD_DOTA':
        write_json(run_dir / 'skipped.json', skipped)
        if skipped:
            print(f'Skipped {len(skipped)} images without a detection; see {run_dir / "skipped.json"}', flush=True)
    build_report(runtime, run_dir, evaluations, rows, summaries, args.save_plots, args.build_html)
    print(f'Explanations: {run_dir}\nEvaluations: {evaluations}')
    from run_timing import report_timing
    timing = report_timing(run_dir, evaluations, image_timings, model_load_sec,
                           time.perf_counter() - run_started, skipped=len(prior), failed=len(failures))
    if failures:
        raise RuntimeError(f'{len(failures)} images failed; see {run_dir / "failures.json"}')
    return {'completed': len(summaries), 'skipped': skipped, 'rows': rows, 'run_dir': run_dir, 'evaluation_dir': evaluations, 'timing': timing}
