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

METHODS = ['BPT', 'SAM_c', 'SAM_c_kernelshap']
DEFAULT_CONFIGS = {'OD_DETR': 'MSCOCO_epito', 'OC_ResNet': 'ImageNet_epito',
                   'OC_ViT': 'ImageNet_epito', 'COD_ViT': 'CAMO_epito'}
MODEL_NAMES = {'OD_DETR': 'facebook/detr-resnet-50', 'OC_ResNet': 'resnet50',
               'OC_ViT': 'vit_b_16', 'COD_ViT': 'vit_b_16'}


def create_parser(kind):
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
    parser.add_argument('--max-evals', type=int, default=1000,
                         help='Maximum number of SAM_c region evaluations per image')
    parser.add_argument('--kernel-shap-samples', type=int)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--kernel-shap-batch-size', type=int)
    parser.add_argument('--model-batch-size', '--resnet-batch-size', '--vit-batch-size', '--detr-batch-size',
                        type=int, default=8, help='Maximum images in one model forward pass')
    parser.add_argument('--auc-batch-size', type=int, default=8)
    parser.add_argument('--auc_num_samples', '--auc-num-samples', type=int, default=101)
    parser.add_argument('--bg-type', choices=['black', 'gray', 'white', 'blurred', 'noise', 'full'], 
                        default='black', help='Background replacement for masked pixels')
    parser.add_argument('--class-aggregation', choices=['sum', 'max'], default='sum')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='DETR scoring threshold (default: 0 for continuous masked scores)')
    parser.add_argument('--detr-shortest-edge', type=int, help='Override the DETR processor resize')
    parser.add_argument('--target-class', type=int, help='Explain this output index instead of the top prediction')
    parser.add_argument('--show_gt', '--show-gt', action='store_true', help='Show/save configured dataset segmentation GT')
    parser.add_argument('--gt-class', help='Optional GT class: face part, ImageNetS synset/label ID, or foreground')
    parser.add_argument('--output-dir', type=Path, help='Override explanation base before bg_*_B*')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--no-auc', dest='compute_auc', action='store_false')
    parser.add_argument('--no-plots', dest='save_plots', action='store_false')
    parser.add_argument('--no-heatmaps', dest='save_heatmaps', action='store_false',
                        help='Skip individual heatmap PNGs; keep combined XAI plots and numeric attributions')
    parser.add_argument('--no-html', dest='build_html', action='store_false')
    parser.add_argument('--verbose-level', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--torch-threads', type=int, help='Optional CPU intra-op thread limit')
    return parser


def normalize_id(value, data):
    value = Path(str(value)).stem
    if data.get('input_format') != 'directory' and value.isdigit():
        return f'{int(value):012d}'
    if data.get('name', '').lower() == 'imagenet' and value.isdigit():
        return f'ILSVRC2012_val_{int(value):08d}'
    return value


def prepare_inputs(kind, args):
    config = load_path_config(args.config)
    data = config['data']
    expected = 'mscoco' if kind == 'OD_DETR' else 'camo' if kind == 'COD_ViT' else 'imagenet'
    if data.get('name', '').replace('_', '').lower() != expected:
        raise ValueError(f'{kind} requires the {expected} dataset config')
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
    for name in ['checkpoint', 'class_names']:
        if getattr(args, name):
            setattr(args, name, resolve_path(getattr(args, name)))
            if not getattr(args, name).exists():
                raise FileNotFoundError(getattr(args, name))
    if kind != 'OD_DETR' and args.model != MODEL_NAMES[kind]:
        raise ValueError(f'{kind} supports --model {MODEL_NAMES[kind]}')
    base = resolve_path(args.output_dir) if args.output_dir else output_path(config, 'explanations') / kind / folder.name
    run_dir = base / f'bg_{args.bg_type}_B{args.max_evals}'
    return {'config': config, 'records': records, 'partition_dir': folder,
            'run_dir': run_dir, 'evaluation_dir': evaluation_path(config, run_dir),
            'annotation_file': annotation_file, 'methods': list(METHODS),
            'ground_truth': data.get('ground_truth') if args.show_gt else None}


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
    ignored = {'limit', 'image_ids', 'dry_run', 'resume', 'verbose_level', 'fail_fast', 'torch_threads', 'output_dir'}
    settings = {key: jsonable(value) for key, value in vars(args).items() if key not in ignored}
    if settings.get('save_heatmaps', True):
        settings.pop('save_heatmaps', None)  # Preserve signatures from before this option existed.
    if settings.get('gt_class') is None:
        settings.pop('gt_class', None)
    if args.show_gt and config and config['data'].get('ground_truth'):
        settings['ground_truth'] = config['data']['ground_truth']
    for name in ['checkpoint', 'class_names']:
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
                     ground_truth=None, bpt_partition=None, auc_results=None):
    plt = runtime.plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    try:
        axes[0, 0].imshow(image)
        for detection in prediction.get('detections', []):
            if detection['class_id'] != prediction['predicted_class_id']:
                continue
            from matplotlib.patches import Rectangle
            x1, y1, x2, y2 = detection['box']
            axes[0, 0].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=1))
        axes[0, 0].set_title('Input')
        for ax, labels, name in [(axes[0, 1], bpt_partition, 'BPT'), (axes[0, 2], partition, 'SAM_c')]:
            if labels is not None:
                ax.imshow(labels, cmap='turbo', interpolation='nearest')
                ax.set_title(f'{name}: {len(np.unique(labels))} regions')
        for ax, method in zip(axes[1], METHODS):
            if method not in attributions:
                continue
            heatmap = np.asarray(attributions[method][0])
            maximum = max(float(np.nanpercentile(np.abs(heatmap), 99.5)), 1e-12)
            handle = ax.imshow(heatmap, cmap='coolwarm', vmin=-maximum, vmax=maximum)
            ax.set_title(f'Attributions-{method}')
            if ground_truth is not None:
                mask = np.asarray(ground_truth['mask'], dtype=bool)
                if mask.shape != heatmap.shape:
                    raise ValueError('GT mask and attribution dimensions must match')
                if mask.any():
                    # Padding also outlines objects touching the image boundary.
                    padded = np.pad(mask.astype(float), 1)
                    ax.contour(np.arange(-1, mask.shape[1] + 1),
                               np.arange(-1, mask.shape[0] + 1), padded,
                               levels=[.5], colors='lime', linewidths=1.3)
                    ax.set_xlim(-.5, mask.shape[1] - .5)
                    ax.set_ylim(mask.shape[0] - .5, -.5)
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            colorbar_ax = divider.append_axes('right', size='4%', pad=.08)
            fig.colorbar(handle, cax=colorbar_ax)
        for ax in axes[:2].ravel():
            ax.axis('off')
        top_classes = prediction.get('top_k_classes') or []
        top = max(top_classes, key=lambda item: float(item['confidence'])) if top_classes else None
        predicted_title = (f"Top predicted: {top['class_name']} ({float(top['confidence']):.4f})"
                           if top else 'Top predicted: unavailable')
        gt_note = (' | GT contour: green' if ground_truth is not None and np.any(ground_truth['mask'])
                   else ' | GT: ' + ground_truth['status'] if ground_truth is not None else '')
        fig.suptitle(f"{predicted_title} | Explained: {prediction['explained_class']}{gt_note}")
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(output, dpi=130)
    finally:
        plt.close(fig)


def build_report(runtime, run_dir, evaluation_dir, rows, summaries, save_plots, build_html):
    pd, plt = runtime.pd, runtime.plt
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv = evaluation_dir / 'auc_results_all_images_current_run.csv'
    frame.to_csv(csv, index=False)
    frame.to_csv(run_dir / csv.name, index=False)
    metrics = [column for column in ['auc_ins', 'auc_del', 'time_exp', 'time_eval',
                                    'sensitivity_n_corr', 'stability_top10_jaccard'] if column in frame]
    if not frame.empty:
        frame.groupby('method', sort=False)[metrics].agg(['mean', 'std', 'median']).to_csv(
            evaluation_dir / 'method_summary.csv')
    if save_plots and not frame.empty:
        fig, axes = plt.subplots(1, max(1, len(metrics)), figsize=(4 * max(1, len(metrics)), 4), squeeze=False)
        try:
            for ax, metric in zip(axes.ravel(), metrics):
                available = [method for method in METHODS if method in set(frame['method'])]
                ax.boxplot([frame.loc[frame['method'] == method, metric].dropna() for method in available],
                           tick_labels=available)
                ax.set_title(metric)
                ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            fig.savefig(evaluation_dir / 'method_boxplots.png', dpi=130)
        finally:
            plt.close(fig)
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
                'table{border-collapse:collapse}td,th{padding:6px}section{margin:24px 0}</style>'
                f'<h1>XAI results</h1><p>{len(summaries)} completed images</p>{table}{"".join(sections)}')
    (evaluation_dir / 'xai_results_report.html').write_text(document)


def run(kind, args, adapter=None):
    inputs = prepare_inputs(kind, args)
    print(json.dumps({key: jsonable(value) for key, value in inputs.items() if key not in {'records', 'config'}}, indent=2))
    print('Images:', ', '.join(record['image_id'] for record in inputs['records']))
    if args.dry_run:
        return inputs
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
    if pending and adapter is None:
        adapter = load_adapter(kind, args, device)
    write_json(run_dir / 'run_info.json', {'runner': kind, 'model': args.model, 'signature': signature,
               'methods': METHODS, 'config': args.config, 'masks_base_path': str(inputs['partition_dir']),
               'image_ids': [record['image_id'] for record in inputs['records']],
               'background_replacement_values': args.bg_type, 'max_evals': args.max_evals,
               'results_dir': run_dir, 'evaluations_dir': evaluations, 'settings': vars(args),
               'ground_truth': inputs['ground_truth'],
               'score_type': 'detection confidence' if kind == 'OD_DETR' else 'classification probability'})
    rows = [row for manifest in prior for row in manifest['rows']]
    summaries = [manifest['prediction'] for manifest in prior]
    failures = []
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
            _, labels = runtime.partitions.load_refine_partitions(image, str(inputs['partition_dir']), image_id, 'compact')
            predict = make_masked_predictor(adapter, image, backgrounds(image, args.bg_type), args.model_batch_size)
            full = predict(np.ones((1, *image.shape[:2]), dtype=bool))[0]
            target = int(np.argmax(full)) if args.target_class is None else args.target_class
            if target < 0 or target >= len(adapter.class_names):
                raise ValueError(f'Target class {target} outside model output range')
            explainer = runtime.shap_bpt.Explainer(predict, image, num_explained_classes=None,
                                                  explained_class_ids=[target], verbose=args.verbose_level == 'high')
            f_s, f_0 = float(explainer.base_nuN[0]), float(explainer.base_nu0[0])
            prediction = {'image_id': image_id, 'source_image': str(record['image_path']),
                          'fixed_category': adapter.class_names[target], 'explained_class': adapter.class_names[target],
                          'predicted_class_id': target, 'f_S': f_s, 'f_0': f_0,
                          'top_k_classes': [{'class_id': int(index), 'class_name': adapter.class_names[int(index)],
                                             'confidence': float(full[index])} for index in np.argsort(full)[::-1][:5]],
                          'detections': adapter.detections(image)}
            ground_truth = None
            if args.show_gt:
                from ground_truth import load_ground_truth, save_ground_truth
                gt_class = (args.gt_class or prediction['explained_class']) if kind == 'OD_DETR' else args.gt_class
                ground_truth = load_ground_truth(
                    inputs['config'], record['image_path'], image.shape, gt_class=gt_class,
                    predicted_index=target, label_space=getattr(adapter, 'label_space', None),
                    predicted_class=prediction['explained_class'])
                prediction['ground_truth'] = save_ground_truth(ground_truth, image, folder, image_id)
            write_json(folder / f'{image_id}_predictions.json', prediction)
            attributions, image_rows, auc_results = {}, [], []
            bpt_labels = None
            for method in METHODS:
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
                        bpt_labels = bpt_partition_labels(tree, len(np.unique(labels)), image.shape[:2])
                    values = explainer.explain_instance(args.max_evals, bpt=tree, method='BPT', batch_size=args.eval_batch_size)
                elapsed = time.perf_counter() - start
                values = np.asarray(values, dtype=float)
                if values.shape != (1, *image.shape[:2]) or not np.isfinite(values).all():
                    raise ValueError(f'{method}: invalid attribution map {values.shape}')
                attributions[method] = values
                row = {'image_id': image_id, 'image_no': image_id, 'method': method,
                       'fixed_category': prediction['explained_class'], 'f_S': f_s, 'f_0': f_0,
                       'time_exp': elapsed, 'time_eval': 0.0}
                if args.compute_auc:
                    result = evaluate(runtime, method, values, predict, target, f_s, f_0, args, elapsed)
                    auc_results.append(result)
                    row.update({key: float(result[key]['auc_clipr']) for key in ['auc_ins', 'auc_del']})
                    row.update({key: value for key, value in result.items()
                                if isinstance(value, (int, float, np.number))})
                image_rows.append(row)
                if args.save_plots and args.save_heatmaps:
                    runtime.utx.plot_single_attributions(values, str(folder), f'{method}_{image_id}.png',
                                                        save_plot=True, destroy_fig=True)
            np.savez_compressed(folder / f'{image_id}_attributions.npz', **attributions)
            write_json(folder / f'{image_id}_metrics.json', auc_results if args.compute_auc else image_rows)
            if args.save_plots:
                explanation_grid(runtime, image, labels, attributions, prediction,
                                 folder / f'{image_id}_xai_results.png', ground_truth, bpt_labels, auc_results)
                if args.compute_auc:
                    runtime.utx.plot_auc_results(auc_results, str(folder), save_plot=True,
                                                 image_no=image_id, destroy_figs=True)
                    runtime.utx.plot_faithfulness_curves(auc_results, str(folder), save_plot=True,
                                                       image_no=image_id, destroy_figs=True)
            artifacts = [f'{image_id}_predictions.json', f'{image_id}_attributions.npz', f'{image_id}_metrics.json']
            if args.save_plots:
                artifacts += [f'{image_id}_xai_results.png']
                if args.save_heatmaps:
                    artifacts += [f'{method}_{image_id}.png' for method in METHODS]
                if args.compute_auc:
                    artifacts += [f'auc_results_{image_id}.png', f'faithfulness_curves_{image_id}.png']
            if args.show_gt:
                artifacts += [f'{image_id}_gt{extension}' for extension in ['.npy', '.png', '.json', '_labels.npz']]
            manifest = {'signature': signature, 'input_stamp': input_stamp(record), 'artifacts': artifacts,
                        'rows': image_rows, 'prediction': prediction, 'total_time_sec': time.perf_counter() - started}
            write_json(marker, manifest)  # Completion marker is always written last.
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
    build_report(runtime, run_dir, evaluations, rows, summaries, args.save_plots, args.build_html)
    print(f'Explanations: {run_dir}\nEvaluations: {evaluations}')
    if failures:
        raise RuntimeError(f'{len(failures)} images failed; see {run_dir / "failures.json"}')
    return {'completed': len(summaries), 'rows': rows, 'run_dir': run_dir, 'evaluation_dir': evaluations}
