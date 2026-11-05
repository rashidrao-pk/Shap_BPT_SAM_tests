#!/usr/bin/env python3
"""Paired AUC-resolution experiment on saved attribution maps; no XAI recomputation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import experiment_common as common


def compare_map(auc, predict, heatmap, target, f_s, f_0, points=(100, 200), reference=1000,
                repeats=3, batch_size=8, synchronize=lambda: None):
    measurements = {(direction, n): [] for direction in ['ins', 'del'] for n in points}
    for repeat in range(repeats):
        # Alternate order to reduce warm-up/order bias between the two resolutions.
        for n in points if repeat % 2 == 0 else points[::-1]:
            for direction in ['ins', 'del']:
                count = [0]
                def counted(masks):
                    count[0] += len(masks)
                    return predict(masks)
                synchronize()
                started = time.perf_counter()
                curve = auc(counted, heatmap, f_s, f_0, target, batch_size=batch_size,
                            method=direction, num_samples=n)
                synchronize()
                measurements[(direction, n)].append((time.perf_counter()-started, curve, count[0]))
    rows, curves = [], {}
    for direction in ['ins', 'del']:
        ref = auc(predict, heatmap, f_s, f_0, target, batch_size=batch_size,
                  method=direction, num_samples=reference)
        curves[f'{direction}_reference'] = common.jsonable(ref)
        for n in points:
            values = measurements[(direction, n)]
            scores = [float(curve['auc_clipr']) for _, curve, _ in values]
            rows.append({'direction': direction, 'requested_points': n,
                'actual_points': int(values[0][1]['evaluated_points']),
                'mask_evaluations': int(values[0][2]),
                'median_seconds': float(np.median([seconds for seconds, _, _ in values])),
                'auc': float(np.median(scores)), 'auc_repeat_range': max(scores)-min(scores),
                'reference_points': reference, 'reference_actual_points': int(ref['evaluated_points']),
                'reference_auc': float(ref['auc_clipr']),
                'absolute_reference_error': abs(float(np.median(scores))-float(ref['auc_clipr']))})
            curves[f'{direction}_{n}'] = common.jsonable(values[0][1])
    return rows, curves


def paired_summary(rows, points, tolerance):
    import pandas as pd
    frame = pd.DataFrame(rows)
    pairs = []
    for (image, method, direction), group in frame.groupby(['image_id', 'method', 'direction'], sort=False):
        low, high = [group[group.requested_points == n].iloc[0] for n in points]
        pairs.append({'image_id': image, 'method': method, 'direction': direction,
            'absolute_auc_change': abs(float(high.auc)-float(low.auc)),
            'error_low': float(low.absolute_reference_error), 'error_high': float(high.absolute_reference_error),
            'seconds_low': float(low.median_seconds), 'seconds_high': float(high.median_seconds),
            'runtime_ratio': float(high.median_seconds/max(low.median_seconds, 1e-12)),
            'masks_low': int(low.mask_evaluations), 'masks_high': int(high.mask_evaluations)})
    paired = pd.DataFrame(pairs)
    ranking_changes = 0
    for _, group in frame.groupby(['image_id', 'direction']):
        pivot = group.pivot(index='method', columns='requested_points', values='auc')
        if len(pivot) > 1 and not pivot[points[0]].rank(method='average').equals(pivot[points[1]].rank(method='average')):
            ranking_changes += 1
    maximum_change = float(paired.absolute_auc_change.max())
    errors = [float(paired.error_low.max()), float(paired.error_high.max())]
    unstable = bool((frame.auc_repeat_range > tolerance).any())
    if unstable:
        recommendation = 'No resolution recommendation: repeated evaluations varied beyond the declared tolerance.'
    elif errors[0] <= tolerance and maximum_change <= tolerance and ranking_changes == 0:
        recommendation = f'Use {points[0]} points for these tested maps: errors and changes are within tolerance, with no method-ranking change.'
    elif errors[1] <= tolerance and errors[0] > tolerance:
        recommendation = f'Use {points[1]} points for these tested maps: it meets the reference tolerance while {points[0]} does not.'
    else:
        recommendation = 'No blanket recommendation: inspect reference errors and method-ranking changes; neither higher resolution nor a changed AUC alone proves greater accuracy.'
    summary = {'points': list(points), 'reference_points': int(frame.reference_points.iloc[0]),
        'tolerance': tolerance, 'image_count': int(frame.image_id.nunique()),
        'method_count': int(frame.method.nunique()), 'max_absolute_auc_change': maximum_change,
        'mean_absolute_auc_change': float(paired.absolute_auc_change.mean()),
        'max_error_low': errors[0], 'max_error_high': errors[1],
        'median_runtime_ratio': float(paired.runtime_ratio.median()),
        'summed_median_seconds_low': float(paired.seconds_low.sum()),
        'summed_median_seconds_high': float(paired.seconds_high.sum()),
        'mean_error_low': float(paired.error_low.mean()), 'mean_error_high': float(paired.error_high.mean()),
        'mask_evaluations_low': int(paired.masks_low.sum()), 'mask_evaluations_high': int(paired.masks_high.sum()),
        'ranking_changed_image_directions': ranking_changes, 'recommendation': recommendation}
    return summary, paired


def restore_model_args(info, device, checkpoint=None):
    runner = info.get('runner') or ('OD_yolo' if info.get('model_group') == 'yolov11' else None)
    if runner not in {*common.DEFAULT_CONFIGS, 'OD_yolo'}:
        raise ValueError('Cannot identify runner from run_info.json')
    args = common.create_parser('OD_DETR' if runner == 'OD_yolo' else runner).parse_args([])
    saved = info.get('settings', {})
    for name in vars(args):
        if name in saved:
            setattr(args, name, saved[name])
        elif name in info:
            setattr(args, name, info[name])
    args.bg_type = info.get('background_replacement_values', args.bg_type)
    args.model = info.get('model', args.model)
    if runner == 'OD_yolo' and info.get('model_source'):
        args.checkpoint = Path(info['model_source'])
    if checkpoint:
        args.checkpoint = checkpoint
    for name in ['checkpoint', 'class_names', 'attribute_file']:
        value = getattr(args, name)
        if value:
            setattr(args, name, common.resolve_path(value))
    args.device = device
    return runner, args


def run(options, adapter=None):
    source = options.run_dir.resolve()
    info = json.loads((source / 'run_info.json').read_text())
    runner, args = restore_model_args(info, options.device, options.checkpoint)
    runtime = common.load_runtime()
    if options.torch_threads:
        runtime.torch.set_num_threads(options.torch_threads)
    device_name = options.device
    if device_name == 'auto':
        device_name = 'cuda' if runtime.torch.cuda.is_available() else 'cpu'
    device = runtime.torch.device(device_name)
    if options.model_batch_size:
        args.model_batch_size = options.model_batch_size
    if options.auc_batch_size:
        args.auc_batch_size = options.auc_batch_size
    files = sorted(source.glob('*/*_attributions.npz'))
    if options.image_ids:
        wanted = set(options.image_ids)
        files = [path for path in files if path.name.removesuffix('_attributions.npz') in wanted]
    files = files[:options.limit]
    if not files:
        raise ValueError('No saved attribution NPZ files found; YOLO requires numeric maps from an --explain_all run')
    evaluation_root = Path(info.get('evaluations_dir', source))
    output = (options.output_dir or evaluation_root / 'auc_point_comparison').resolve()
    if output == source:
        raise ValueError('Comparison output must be separate from the source experiment')
    if adapter is None:
        if runner == 'OD_yolo':
            from budget_ablation import YoloAdapter
            adapter = YoloAdapter(args, device)
        else:
            adapter = common.load_adapter(runner, args, device)
    synchronize = (lambda: runtime.torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    common.write_json(output / 'comparison_status.json', {'state': 'running', 'source_run': str(source),
        'points': options.points, 'reference_points': options.reference_points})
    for name in ['summary.json', 'conclusion.md']:
        (output / name).unlink(missing_ok=True)
    rows = []
    for archive in runtime.tqdm(files, desc='AUC point comparison', ascii=True):
        image_id = archive.name.removesuffix('_attributions.npz')
        prediction = json.loads((archive.parent / f'{image_id}_predictions.json').read_text())
        image_path = prediction.get('source_image')
        if image_path is None:
            image_path = str(Path(info['image_dir']) / f"{int(prediction['image_id']):012d}.jpg")
        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert('RGB'))
        if runner == 'OD_yolo':
            # Preserve legacy YOLO's exact blur/noise/background construction.
            import OD_yolo as od
            od.load_runtime_dependencies()
            image, _, replacements, _ = od.load_image(image_path, device, args.bg_type)
            predict = od.make_predict_yolo_masked(adapter.model, image, replacements, args.class_aggregation)
        else:
            replacements = common.backgrounds(image, args.bg_type)
            predict = common.make_masked_predictor(adapter, image, replacements, args.model_batch_size)
        target = prediction.get('predicted_class_id')
        if target is None:
            target = adapter.class_names.index(prediction['explained_class'])
        endpoints = predict(np.stack([np.ones(image.shape[:2], dtype=bool), np.zeros(image.shape[:2], dtype=bool)]))[:, target]
        if not np.allclose(endpoints, [prediction['f_S'], prediction['f_0']], atol=options.score_tolerance, rtol=1e-3):
            raise ValueError(f'{image_id}: model/background scores differ from the saved run; verify checkpoint and settings')
        if abs(endpoints[0] - endpoints[1]) <= 1e-12:
            raise ValueError(f'{image_id}: equal full/background scores make normalized AUC uninformative')
        synchronize()  # Endpoint checks also warm the model; excluded from timed curves.
        with np.load(archive, allow_pickle=False) as maps:
            methods = options.methods or list(maps.files)
            if not methods:
                raise ValueError(f'No saved attribution maps in {archive}')
            missing = set(methods) - set(maps.files)
            if missing:
                raise ValueError(f'{archive}: requested methods are not saved: {sorted(missing)}; available: {maps.files}')
            for method in methods:
                values = np.asarray(maps[method], dtype=float)
                if values.shape != (1, *image.shape[:2]) or not np.isfinite(values).all():
                    raise ValueError(f'{image_id}/{method}: invalid saved attribution map')
                measured, curves = compare_map(runtime.utx.saliency_to_auc, predict, values[0], target,
                    *endpoints, tuple(options.points), options.reference_points, options.repeats,
                    args.auc_batch_size, synchronize)
                for row in measured:
                    row.update(image_id=image_id, method=method, background_count=len(replacements),
                               model_images=row['mask_evaluations'] * len(replacements))
                rows.extend(measured)
                print(f'{image_id}: {method} compared', flush=True)
                common.write_json(output / 'curves' / image_id / f'{method}.json', curves)
        runtime.pd.DataFrame(rows).to_csv(output / 'measurements.csv', index=False)
    summary, paired = paired_summary(rows, options.points, options.tolerance)
    paired.to_csv(output / 'paired_comparisons.csv', index=False)
    frame = runtime.pd.DataFrame(rows)
    fig, axes = runtime.plt.subplots(1, 3, figsize=(16, 5))
    try:
        for ax, (metric, aggregate, title) in zip(axes, [
                ('actual_points', 'mean', 'Mean evaluated points per curve'),
                ('median_seconds', 'median', 'Median seconds per curve'),
                ('absolute_reference_error', 'max', 'Worst AUC error vs reference')]):
            table = frame.groupby(['method', 'requested_points'])[metric].agg(aggregate).unstack()
            table.plot.bar(ax=ax, rot=45)
            ax.set_title(title); ax.set_xlabel('Method'); ax.grid(axis='y', alpha=.2)
        fig.tight_layout(); fig.savefig(output / 'auc_resolution_comparison.png', dpi=150)
    finally:
        runtime.plt.close(fig)
    common.write_json(output / 'summary.json', {**summary, 'source_run': str(source), 'runner': runner,
        'repeats': options.repeats, 'settings': vars(args), 'device': str(device)})
    text = ('# AUC point comparison\n\n' + summary['recommendation'] + '\n\n'
        f"- Maximum absolute AUC change: {summary['max_absolute_auc_change']:.6f}\n"
        f"- Maximum errors vs {options.reference_points}-point reference: "
        f"{summary['max_error_low']:.6f} ({options.points[0]}), {summary['max_error_high']:.6f} ({options.points[1]})\n"
        f"- Median {options.points[1]}/{options.points[0]} runtime ratio: {summary['median_runtime_ratio']:.2f}x\n"
        f"- Sum of median curve times: {summary['summed_median_seconds_low']:.3f}s vs {summary['summed_median_seconds_high']:.3f}s\n"
        f"- Mask evaluations per repeat: {summary['mask_evaluations_low']} vs {summary['mask_evaluations_high']}\n"
        f"- Image/direction groups with changed method ranking: {summary['ranking_changed_image_directions']}\n\n"
        'These results apply only to the selected saved maps/model/background. The denser reference is approximate. '
        'Timings exclude model loading, explanation generation, warm-up endpoints, and reference evaluation. '
        'Larger AUC values are not inherently better: insertion and deletion have different interpretations.\n')
    (output / 'conclusion.md').write_text(text)
    common.write_json(output / 'comparison_status.json', {'state': 'complete', 'source_run': str(source),
        'points': options.points, 'reference_points': options.reference_points})
    print(text)
    print(f'Report: {output}')
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path, help='Existing XAI experiment directory containing run_info.json')
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--image-ids', nargs='+', help='Exact saved attribution filename IDs')
    parser.add_argument('--methods', nargs='+', help='Exact keys from the saved attribution NPZ; default all saved methods')
    parser.add_argument('--points', nargs=2, type=int, default=[100, 200])
    parser.add_argument('--reference-points', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tolerance', type=float, default=.005, help='Predeclared absolute normalized-AUC tolerance')
    parser.add_argument('--score-tolerance', type=float, default=1e-4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--checkpoint', type=Path, help='Override the saved model checkpoint location')
    parser.add_argument('--model-batch-size', type=int)
    parser.add_argument('--auc-batch-size', type=int)
    parser.add_argument('--torch-threads', type=int)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if (args.limit < 1 or args.repeats < 2 or args.points[0] < 2 or args.points[0] >= args.points[1]
            or args.reference_points <= args.points[1] or args.tolerance <= 0 or args.score_tolerance < 0):
        parser.error('Use positive limits/tolerances, at least two repeats, and 2 <= low < high < reference points')
    for name in ['model_batch_size', 'auc_batch_size', 'torch_threads']:
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    return args


if __name__ == '__main__':
    run(parse_args())
