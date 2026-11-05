#!/usr/bin/env python3
"""Small, reproducible budget pilot; choose on calibration and validate on held-out images."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

import experiment_common as common
from budget_selection import DEFAULT_CRITERIA, compare_maps, passes, select_budget, load_study
from config_paths import output_path, resolve_path
from auc_export import auc_columns

RUNNERS = ['OC_ResNet', 'OC_VGG', 'OC_ViT', 'COD_ViT', 'OD_DETR', 'OD_yolo',
           'OC_CelebA', 'OC_BUSI', 'OC_PartImageNet']
METHODS = ['BPT', 'BPT+SAM_c', 'SAM_c+KernelSHAP']


def parser(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--runner', choices=RUNNERS, default='OC_ResNet')
    runner = preliminary.parse_known_args(argv)[0].runner
    result = common.create_parser('OD_DETR' if runner == 'OD_yolo' else runner)
    if runner == 'OD_yolo':
        result.set_defaults(model='yolo11s')
    result.description = __doc__
    result.add_argument('--runner', choices=RUNNERS, default=runner)
    result.add_argument('--budgets', nargs='+', type=int, default=[100, 200, 500, 1000, 2000])
    result.add_argument('--reference-budget', type=int, default=4000)
    result.add_argument('--stop-after-images', type=int, help='Save an interim report after the first N fixed pilot images')
    result.add_argument('--repeats', type=int, default=3)
    result.add_argument('--pilot-seed', type=int, default=0)
    result.add_argument('--holdout-fraction', type=float, default=.4)
    result.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    result.add_argument('--study-name', default='pilot')
    result.add_argument('--min-spearman', type=float, default=DEFAULT_CRITERIA['min_spearman'])
    result.add_argument('--min-top-jaccard', type=float, default=DEFAULT_CRITERIA['min_top_jaccard'])
    result.add_argument('--max-relative-l1', type=float, default=DEFAULT_CRITERIA['max_relative_l1'])
    result.add_argument('--max-auc-difference', type=float, default=DEFAULT_CRITERIA['max_auc_difference'])
    result.add_argument('--min-image-pass-fraction', type=float, default=1.)
    result.add_argument('--combine', nargs='+', type=Path, help='Combine completed pilot folders, without model inference')
    result.set_defaults(limit=5, bg_type='full')
    return result.parse_args(argv)


def validate(args):
    args.budgets = sorted(set(args.budgets))
    args.methods = list(dict.fromkeys(args.methods))
    if not args.budgets or min(args.budgets) < 2 or args.reference_budget <= max(args.budgets):
        raise ValueError('Use budgets >=2 and a reference budget strictly above every candidate')
    if args.repeats < 2:
        raise ValueError('Use at least 2 repeats to assess reference uncertainty')
    if args.stop_after_images is not None and args.stop_after_images < 1:
        raise ValueError('--stop-after-images must be positive')
    if not 0 < args.holdout_fraction < 1:
        raise ValueError('--holdout-fraction must be between zero and one')
    if args.limit is not None and args.limit < 3:
        raise ValueError('Use at least 3 pilot images: at least 2 calibration and 1 held-out')
    if not args.study_name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('--study-name must contain letters, numbers, underscores or hyphens')
    for name in ['min_spearman', 'min_top_jaccard', 'min_image_pass_fraction']:
        if not 0 < getattr(args, name) <= 1:
            raise ValueError(f'{name} must be in (0,1]')
    if args.max_relative_l1 < 0 or args.max_auc_difference < 0:
        raise ValueError('Error tolerances must be nonnegative')
    if args.explain_all or args.budget_selection is not None:
        raise ValueError('This is the separate pilot; do not pass --explain_all or --budget-selection')


def choose_examples(records, count, seed, holdout_fraction):
    if len(records) < 3:
        raise ValueError('Fewer than 3 eligible images; generate more partitions before running the pilot')
    count = min(count or 5, len(records))
    if count < 3:
        raise ValueError('A pilot needs at least 3 examples')
    order = np.random.default_rng(seed).permutation(len(records))[:count]
    chosen = [records[int(index)] for index in order]
    held_out = min(count-2, max(1, int(round(count*holdout_fraction))))
    splits = {'calibration': [record['image_id'] for record in chosen[:-held_out]],
              'validation': [record['image_id'] for record in chosen[-held_out:]]}
    return chosen, splits


class YoloAdapter:
    """Same post-NMS detector class scores as the original YOLO runner."""
    def __init__(self, args, device):
        import OD_yolo as od
        od.load_runtime_dependencies()
        self.od, self.batch_size, self.aggregate = od, args.model_batch_size, args.class_aggregation
        source = od.resolve_yolo_source(common.ROOT, args.model,
                                        weights=str(args.checkpoint) if args.checkpoint else None)
        self.model = od.YOLO(source).to(device)
        self.device = 0 if device.type == 'cuda' else str(device)
        names = self.model.names
        self.class_names = [names[i] for i in range(len(names))]
        self.name = str(source)

    def predict_batch(self, images):
        rows = []
        for start in range(0, len(images), self.batch_size):
            results = self.model.predict(list(images[start:start+self.batch_size]), verbose=False, device=self.device)
            rows.extend(self.od.yolo_result_to_scores(result, len(self.class_names), self.aggregate) for result in results)
        return np.asarray(rows)


@contextmanager
def fixed_seed(torch, seed):
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(seed)
            yield
    finally:
        np.random.set_state(state)


def auc_summary(runtime, predict, values, target, f_s, f_0, args):
    if not args.compute_auc:
        return {}
    result = {}
    for direction in ['ins', 'del']:
        curve = runtime.utx.saliency_to_auc(predict, values, f_s, f_0, target,
                                            batch_size=args.auc_batch_size, method=direction,
                                            num_samples=args.auc_num_samples)
        result.update(auc_columns({'auc_'+direction: curve}))
        result['auc_'+direction+'_points'] = len(curve['xs'])
    return result


def evaluate_map(runtime, args, image, labels, predict, explainer, target, trees,
                 method, budget, seed, cache, signature, counters):
    marker = cache.with_suffix('.json')
    archive = cache.with_suffix('.npz')
    if args.resume and marker.is_file() and archive.is_file():
        saved = json.loads(marker.read_text())
        if saved.get('signature') == signature:
            with np.load(archive, allow_pickle=False) as source:
                values = source['values']
            if values.shape == image.shape[:2] and np.isfinite(values).all():
                return values, saved
    print(f"  {image.shape[1]}x{image.shape[0]} {method}: B={budget}, seed={seed}", flush=True)
    started, before = time.perf_counter(), counters['masks']
    with fixed_seed(runtime.torch, seed):
        if method == 'SAM_c+KernelSHAP':
            if len(np.unique(labels)) == 1:
                values = np.full((1, *labels.shape), float(explainer.base_nuN[0])-float(explainer.base_nu0[0]))
            else:
                values = runtime.utx.kernel_shap_partition_regions(
                    predict, labels, target, n_samples=budget,
                    batch_size=args.kernel_shap_batch_size or args.eval_batch_size, seed=seed)
        else:
            values = explainer.explain_instance(budget, bpt=trees[method], method='BPT', batch_size=args.eval_batch_size)
    seconds = time.perf_counter()-started
    mask_count = counters['masks']-before
    values = np.asarray(values, dtype=float)[0]
    if values.shape != image.shape[:2] or not np.isfinite(values).all():
        raise ValueError('Invalid pilot attribution map')
    evaluation_started = time.perf_counter()
    auc = auc_summary(runtime, predict, values, target, float(explainer.base_nuN[0]), float(explainer.base_nu0[0]), args)
    saved = {'signature': signature, 'method': method, 'budget': budget, 'seed': seed,
             'explanation_sec': seconds, 'evaluation_sec': time.perf_counter()-evaluation_started,
             'mask_evaluations': mask_count, 'model_images': mask_count*counters['backgrounds'], **auc}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, values=values)
    common.write_json(marker, saved)  # Written last, so interrupted work is not resumed as complete.
    return values, saved


def differences(values, reference, auc, reference_auc):
    result = compare_maps(values, reference)
    for direction in ['ins', 'del']:
        key = 'auc_'+direction
        result[key+'_difference'] = abs(auc[key]-reference_auc[key]) if key in auc and key in reference_auc else None
    return result


def plot_image(runtime, folder, image, labels, means, references, args, ground_truth):
    from attribution_visuals import notebook_style, draw_gt_boundaries
    fig, axes = runtime.plt.subplots(len(args.methods), len(args.budgets)+2,
                                    figsize=(2.3*(len(args.budgets)+2), 2.6*len(args.methods)), squeeze=False)
    try:
        for row, method in enumerate(args.methods):
            axes[row, 0].imshow(image)
            axes[row, 0].set_title('Input')
            axes[row, 0].set_ylabel(method)
            style = notebook_style(references[method])  # Same color scale across each budget row.
            for column, budget in enumerate([*args.budgets, args.reference_budget], 1):
                values = references[method] if budget == args.reference_budget else means[(method, budget)]
                axes[row, column].imshow(values, **style)
                axes[row, column].set_title(f'{"Reference " if budget == args.reference_budget else ""}B={budget}')
                if ground_truth is not None and np.any(ground_truth['mask']):
                    draw_gt_boundaries(axes[row, column], ground_truth['mask'])
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle('Budget ablation — repeat means; common reference color scale within each method')
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(folder/'budget_heatmaps.png', dpi=130)
    finally:
        runtime.plt.close(fig)


def budget_conclusion(selection, studies, selection_path):
    """Describe the observed decision without recommending a failed or partial pilot."""
    status = selection['status']
    budget = selection.get('selected_budget')
    candidate = selection.get('calibration_candidate')
    all_methods = set(selection['methods']) == set(METHODS)
    if status == 'validated' and all_methods:
        summary = (f'Recommended budget: {budget} for BPT, BPT+SAM_c, and SAM_c+KernelSHAP. '
                   'This is the smallest tested common budget that passed calibration and held-out '
                   'validation for every included pilot under the declared criteria.')
    elif status == 'validated':
        summary = (f'Budget {budget} passed for the tested methods only. No common budget for all '
                   'three partition methods is recommended because this ablation omitted methods.')
    elif status == 'validation_failed':
        summary = (f'No budget is recommended. Calibration selected {candidate}, but it failed '
                   'held-out validation. Do not substitute a larger budget using these same '
                   'held-out outcomes; revise the study with fresh validation images.')
    elif status == 'stability_only':
        summary = (f'No production budget is recommended. Budget {candidate} passed map-stability '
                   'checks, but AUC was disabled. Repeat the declared study with AUC enabled '
                   'before applying a budget selection.')
    else:
        summary = ('No tested common budget is recommended. No candidate passed the calibration '
                   'criteria across all included pilots and methods. Inspect failed metrics and '
                   'reference stability before designing a new study; do not assume the largest '
                   'tested budget or reference budget is adequate.')
    lines = ['# Budget ablation conclusion', '', summary, '',
             f"Status: `{status}`. Tested budgets: {', '.join(map(str, selection['tested_budgets']))}. "
             f"Reference budget: {selection['reference_budget']}.", '',
             '## Scope of the evidence', '',
             '| Dataset | Runner | Partitions | Background | Calibration images | Validation images |',
             '| --- | --- | --- | --- | --- | --- |']
    for study in studies:
        stamps = study.get('inputs', [])
        partition = (Path(stamps[0]['partition_path'][0]).parent.name
                     if stamps and stamps[0].get('partition_path') else 'not recorded')
        lines.append(f"| {study['dataset']} | {study['runner']} | {partition} | "
                     f"{study['background']} | {len(study['splits']['calibration'])} | "
                     f"{len(study['splits']['validation'])} |")
    lines += ['', 'Methods tested: ' + ', '.join(selection['methods']) + '.',
              f"AUC: {selection['auc_points']} requested points per curve; "
              + ('included in selection.' if selection['auc_checked'] else 'not checked.'), '']
    if status == 'validated' and all_methods:
        lines += ['## Use this budget', '',
                  f'Pass `--budget-selection {selection_path}` to the production runner. '
                  'It supplies one budget to all three partition-based methods. Keep the same '
                  'model/checkpoint, partition setup, background, class scoring, and AUC resolution.', '']
    elif candidate is not None and selection.get('validation'):
        failed = [group for group in selection['validation'] if not group['pass']]
        if failed:
            lines += ['Failed held-out groups:', '']
            for group in failed:
                lines.append(f"- Pilot {group['study'] + 1}, {group['runner']}, {group['method']}: "
                             f"{group['passed_images']}/{group['images']} images passed.")
            lines.append('')
    unstable = sum(not row.get('reference_stable', False) for study in studies for row in study['rows'])
    if unstable:
        lines += [f'{unstable} comparison rows were marked as having an unstable reference. '
                  'These rows cannot support convergence.', '']
    lines += ['This conclusion is limited to the tested grid, examples, and scopes above. '
              'A single pilot does not establish a budget for untested models, partition variants, '
              'backgrounds, or datasets. Combine all intended pilots before claiming a common budget. '
              'The high-budget reference is approximate, not exact Shapley ground truth.', '']
    return summary, '\n'.join(lines)


def write_reports(folder, studies, save_plots=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    selection = select_budget(studies)
    rows = [dict(row, runner=study['runner'], dataset=study['dataset'], attribute=study.get('attribute'), study=index)
            for index, study in enumerate(studies) for row in study['rows']]
    frame = pd.DataFrame(rows)
    for key in ['spearman', 'top_jaccard', 'relative_l1', 'auc_ins_difference', 'auc_del_difference',
                'explanation_sec', 'mask_evaluations']:
        frame[key] = pd.to_numeric(frame[key], errors='coerce')
    frame.to_csv(folder/'budget_comparisons.csv', index=False)
    common.write_json(folder/'selection.json', selection)
    conclusion_summary, conclusion = budget_conclusion(selection, studies, folder/'selection.json')
    (folder/'conclusion.md').write_text(conclusion)
    grouped = frame.groupby(['runner', 'method', 'budget'], sort=False)
    grouped.agg({'spearman': ['min', 'median'], 'top_jaccard': ['min', 'median'],
                 'relative_l1': ['max', 'median'], 'explanation_sec': ['median', 'max'],
                 'mask_evaluations': ['median', 'max']}).to_csv(folder/'budget_summary.csv')
    specs = [('spearman', 'min', 'Worst rank agreement', 'min_spearman'),
             ('top_jaccard', 'min', 'Worst top-10% overlap', 'min_top_jaccard'),
             ('relative_l1', 'max', 'Worst relative L1 error', 'max_relative_l1'),
             ('auc_ins_difference', 'max', 'Worst insertion AUC difference', 'max_auc_difference'),
             ('auc_del_difference', 'max', 'Worst deletion AUC difference', 'max_auc_difference'),
             ('explanation_sec', 'median', 'Median explanation seconds', None)]
    if save_plots:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        try:
            for ax, (key, aggregate, title, threshold) in zip(axes.ravel(), specs):
                for (runner, method), group in frame.groupby(['runner', 'method'], sort=False):
                    series = group.groupby('budget')[key].agg(aggregate)
                    ax.plot(series.index, series.values, marker='o', label=f'{runner}: {method}')
                if threshold:
                    ax.axhline(studies[0]['criteria'][threshold], linestyle='--', color='black', alpha=.5)
                ax.set_title(title); ax.set_xlabel('Requested budget'); ax.grid(alpha=.2)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(f"Calibration candidate: {selection['calibration_candidate']} | {selection['status']}")
            fig.tight_layout(); fig.savefig(folder/'budget_curves.png', dpi=140)
        finally:
            plt.close(fig)
    images = ''.join(f'<figure><figcaption>{html.escape(str(path.parent.name))}</figcaption>'
                     f'<img src="{html.escape(str(path.relative_to(folder)))}"></figure>'
                     for path in sorted(folder.glob('*/budget_heatmaps.png')))
    document = ('<!doctype html><meta charset="utf-8"><title>Budget ablation</title>'
                '<style>body{font:15px sans-serif;margin:24px}img{max-width:100%}td,th{padding:6px}</style>'
                f'<h1>Budget ablation: {html.escape(selection["status"])}</h1>'
                f'<h2>Conclusion</h2><p>{html.escape(conclusion_summary)}</p>'
                '<p><a href="conclusion.md">Read the complete conclusion and scope</a></p>'
                f'<p>{html.escape(selection["claim"])}</p>'
                f'<p>Calibration candidate: {selection["calibration_candidate"]}; validated budget: {selection["selected_budget"]}</p>'
                '<p>GT best-IoU values are oracle-threshold diagnostics only, not selection criteria.</p>'
                + ('<img src="budget_curves.png">' if save_plots else '') + frame.to_html(index=False, escape=True) + images)
    (folder/'budget_report.html').write_text(document)
    print(f"Budget selection: {selection['status']}; selected={selection['selected_budget']}; calibration candidate={selection['calibration_candidate']}")
    print('Conclusion: ' + conclusion_summary)
    print(f'Conclusion saved: {folder / "conclusion.md"}')
    print(f'Report: {folder / "budget_report.html"}')
    return selection


def run(args, adapter=None):
    started = time.perf_counter()
    if args.combine:
        if args.output_dir is None:
            raise ValueError('--combine requires --output-dir for the common selection report')
        folder = resolve_path(args.output_dir); folder.mkdir(parents=True, exist_ok=True)
        studies = [load_study(path) for path in dict.fromkeys(args.combine)]
        return write_reports(folder, studies, save_plots=args.save_plots)
    validate(args)
    count, args.limit = args.limit, None  # Select uniformly from eligible partitions, not the first N filenames.
    inputs = common.prepare_inputs('OD_DETR' if args.runner == 'OD_yolo' else args.runner, args)
    args.limit = count
    records, splits = choose_examples(inputs['records'], count, args.pilot_seed, args.holdout_fraction)
    criteria = {key: getattr(args, key) for key in DEFAULT_CRITERIA}
    study = {'version': 1, 'state': 'running', 'runner': args.runner, 'dataset': inputs['config']['data']['name'],
             'background': args.bg_type, 'attribute': (args.attribute or ('Brown_Hair' if args.target_class is None else f'class_{args.target_class}')) if args.runner == 'OC_CelebA' else None,
             'budgets': args.budgets, 'reference_budget': args.reference_budget,
             'methods': args.methods, 'repeats': args.repeats, 'pilot_seed': args.pilot_seed,
             'auc_points': args.auc_num_samples, 'compute_auc': args.compute_auc, 'criteria': criteria,
             'splits': splits, 'eligible_images': len(inputs['records']),
             'settings': common.jsonable(vars(args)), 'inputs': [common.input_stamp(record) for record in records]}
    from importlib.metadata import version, PackageNotFoundError
    study['versions'] = {}
    for package in ['torch', 'torchvision', 'shap-bpt', 'numpy', 'scipy']:
        try:
            study['versions'][package] = version(package)
        except PackageNotFoundError:
            study['versions'][package] = 'unknown'
    study['model_files'] = {name: [Path(value).stat().st_size, Path(value).stat().st_mtime_ns]
                            for name in ['checkpoint', 'class_names', 'attribute_file']
                            if (value := getattr(args, name)) is not None}
    identity = {key: value for key, value in study.items() if key not in {'state'}}
    for key in ['resume', 'dry_run', 'verbose_level', 'torch_threads', 'fail_fast', 'stop_after_images']:
        identity['settings'].pop(key, None)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    study['signature'] = signature
    root = resolve_path(args.output_dir) if args.output_dir else output_path(inputs['config'], 'evaluations')/'budget_ablation'
    folder = root/args.runner/inputs['partition_dir'].name/f'bg_{args.bg_type}'
    if args.runner == 'OC_CelebA':
        folder = folder/(study['attribute'] or f'class_{args.target_class}')
    folder = folder/args.study_name
    print(json.dumps({'folder': str(folder), 'eligible': study['eligible_images'], 'splits': splits,
                      'budgets': args.budgets, 'reference_budget': args.reference_budget,
                      'repeats': args.repeats, 'criteria': criteria}, indent=2))
    if args.dry_run:
        return study
    if (folder/'study.json').is_file():
        previous = json.loads((folder/'study.json').read_text())
        if previous.get('signature') != signature:
            raise ValueError('Pilot settings/inputs changed; use a different --study-name')
    folder.mkdir(parents=True, exist_ok=True)
    common.write_json(folder/'study.json', study)  # Image split is fixed before loading the model or looking at outcomes.
    runtime = common.load_runtime()
    if args.torch_threads:
        runtime.torch.set_num_threads(args.torch_threads)
    device = args.device or inputs['config'].get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if runtime.torch.cuda.is_available() else 'mps' if runtime.torch.backends.mps.is_available() else 'cpu'
    device = runtime.torch.device(device)
    study['hardware'] = {'device': str(device), 'gpu': runtime.torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    model_started = time.perf_counter()
    adapter = adapter or (YoloAdapter(args, device) if args.runner == 'OD_yolo' else common.load_adapter(args.runner, args, device))
    study['model_load_sec'] = time.perf_counter()-model_started
    rows, diagnostics = [], []
    split_for = {image: split for split, ids in splits.items() for image in ids}
    run_records = records[:args.stop_after_images] if args.stop_after_images else records
    study['execution_limit'] = len(run_records)
    for record in runtime.tqdm(run_records, desc='Budget pilot', ascii=True):
        image_id = record['image_id']; image_folder = folder/image_id; image_folder.mkdir(exist_ok=True)
        image = np.asarray(Image.open(record['image_path']).convert('RGB'))
        _, labels = runtime.partitions.load_refine_partitions(image, str(inputs['partition_dir']), image_id, 'compact')
        replacements = common.backgrounds(image, args.bg_type)
        base_predict = common.make_masked_predictor(adapter, image, replacements, args.model_batch_size)
        counters = {'masks': 0, 'backgrounds': len(replacements)}
        def predict(masks):
            counters['masks'] += len(masks)
            return base_predict(masks)
        full = predict(np.ones((1, *image.shape[:2]), dtype=bool))[0]
        target = (adapter.class_names.index(args.attribute or 'Brown_Hair') if args.runner == 'OC_CelebA' and args.target_class is None
                  else int(np.argmax(full)) if args.target_class is None else args.target_class)
        if not 0 <= target < len(adapter.class_names):
            raise ValueError('Target class outside classifier output range')
        explainer = runtime.shap_bpt.Explainer(predict, image, num_explained_classes=None, explained_class_ids=[target], verbose=False)
        tree_started = time.perf_counter()
        trees = {method: runtime.shap_bpt.build_bpt_from_image(image, **({'prebuilt_partitions': labels} if method == 'BPT+SAM_c' else {}))
                 for method in args.methods if method != 'SAM_c+KernelSHAP'}
        tree_seconds = time.perf_counter()-tree_started
        prediction = {'image_id': image_id, 'target': target, 'class_name': adapter.class_names[target],
                      'f_S': float(explainer.base_nuN[0]), 'f_0': float(explainer.base_nu0[0]),
                      'tree_build_sec': tree_seconds}
        common.write_json(image_folder/'target.json', prediction)
        gt = None
        if args.show_gt:
            from ground_truth import load_ground_truth, save_ground_truth
            gt = load_ground_truth(inputs['config'], record['image_path'], image.shape,
                                   gt_class=(args.gt_class or prediction['class_name']) if args.runner in {'OD_DETR','OD_yolo'} else args.gt_class,
                                   predicted_index=target, label_space=getattr(adapter, 'label_space', None),
                                   predicted_class=prediction['class_name'])
            save_ground_truth(gt, image, image_folder, image_id)
        references, means = {}, {}
        for method in args.methods:
            cache_dir = image_folder/method.replace('+','_')
            ref_maps, ref_stats = [], []
            for repeat in range(args.repeats):
                values, stat = evaluate_map(runtime, args, image, labels, predict, explainer, target, trees,
                                            method, args.reference_budget, args.pilot_seed+100000+repeat,
                                            cache_dir/f'reference_s{repeat}', signature, counters)
                ref_maps.append(values); ref_stats.append(stat)
            reference = np.mean(ref_maps, axis=0); references[method] = reference
            ref_auc = auc_summary(runtime, predict, reference, target, prediction['f_S'], prediction['f_0'], args)
            ref_diagnostics = []
            for repeat, values in enumerate(ref_maps):
                other = np.mean([value for index, value in enumerate(ref_maps) if index != repeat], axis=0)
                metrics = compare_maps(values, other)
                # Reference uncertainty is checked on maps; candidate AUC uses the repeat-mean map.
                ref_diagnostics.append({**metrics, 'reference_stable': True})
            stable = all(passes(value, criteria, use_auc=False) for value in ref_diagnostics)
            diagnostics.append({'image_id': image_id, 'method': method, 'stable': stable, 'repeats': ref_diagnostics})
            np.savez_compressed(cache_dir/'reference_mean.npz', values=reference, std=np.std(ref_maps,axis=0))
            for budget in args.budgets:
                maps = []
                for repeat in range(args.repeats):
                    values, stat = evaluate_map(runtime, args, image, labels, predict, explainer, target, trees,
                                                method, budget, args.pilot_seed+repeat,
                                                cache_dir/f'B{budget}_s{repeat}', signature, counters)
                    maps.append(values)
                    row = {'image_id': image_id, 'split': split_for[image_id], 'method': method,
                           'budget': budget, 'repeat': repeat, 'reference_stable': stable,
                           **{key: value for key, value in stat.items() if key not in {'signature','budget','method'}},
                           **differences(values, reference, stat, ref_auc)}
                    if gt is not None:
                        from all_methods import best_iou
                        iou = best_iou(values, gt)
                        row['best_iou_oracle'] = iou['best_iou'] if iou else None
                    rows.append(row)
                means[(method, budget)] = np.mean(maps, axis=0)
            common.write_json(folder/'comparisons.json', rows)
            common.write_json(folder/'reference_diagnostics.json', diagnostics)
        if args.save_plots:
            plot_image(runtime, image_folder, image, labels, means, references, args, gt)
    if len(run_records) < len(records):
        from budget_interim import write_interim
        study.update(state='interim', completed_images=[record['image_id'] for record in run_records],
                     elapsed_sec=time.perf_counter()-started)
        common.write_json(folder/'study.json', study)
        common.write_json(folder/'comparisons.json', rows)
        return write_interim(folder, study, rows, diagnostics)
    study.update(state='complete', elapsed_sec=time.perf_counter()-started)
    common.write_json(folder/'study.json', study)
    common.write_json(folder/'comparisons.json', rows)
    study['rows'] = rows
    return write_reports(folder, [study], save_plots=args.save_plots)


if __name__ == '__main__':
    run(parser())
