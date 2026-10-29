"""Optional baseline comparison, with explicit method status and GT IoU panels.

References and method conventions are documented in docs/XAI/ALL_METHODS.md.
Imports of optional attribution packages are deferred until comparison mode.
"""
from contextlib import contextmanager
import copy
import importlib.util
import time
import textwrap
from types import MethodType

import numpy as np


def add_arguments(parser):
    parser.add_argument('--explain_all', '--explain-all', action='store_true',
                        help='Add budgeted BPT/AA/LIME and gradient baseline comparisons')
    parser.add_argument('--bpt-budgets', nargs='+', type=int, default=[100, 500, 1000])
    parser.add_argument('--lime-budgets', nargs='+', type=int, default=[50, 100, 200],
                        help='LIME perturbation samples, not superpixel counts')
    parser.add_argument('--gradient-samples', type=int, default=20)
    parser.add_argument('--idg-steps', type=int, default=50)
    parser.add_argument('--attribution-batch-size', type=int, default=4)
    parser.add_argument('--explanation-seed', type=int, default=0)


def validate(args):
    for name in ['bpt_budgets', 'lime_budgets']:
        if not getattr(args, name) or any(value < 2 for value in getattr(args, name)):
            raise ValueError(f'{name} must contain budgets of at least 2')
    for name in ['gradient_samples', 'attribution_batch_size']:
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.idg_steps < 2:
        raise ValueError('--idg-steps must be at least 2')
    missing = [module for module in ['captum', 'lime', 'shap'] if importlib.util.find_spec(module) is None]
    if missing:
        raise ImportError('Install comparison dependencies: python -m pip install -r requirements-xai-all.txt '
                          f'(missing: {", ".join(missing)})')


def method_names(args):
    return ([f'BPT-{budget}' for budget in dict.fromkeys(args.bpt_budgets)] +
            [f'AA-{budget}' for budget in dict.fromkeys(args.bpt_budgets)] +
            [f'LIME-{budget}' for budget in dict.fromkeys(args.lime_budgets)] +
            ['LRP', 'GradCAM', 'IDG', 'GradExp', 'GradShap'])


def comparison_signature(args):
    import hashlib
    import json
    ignored = {'limit', 'image_ids', 'dry_run', 'resume', 'verbose', 'verbose_level', 'verbose_k', 'fail_fast'}
    values = {key: value for key, value in vars(args).items() if key not in ignored}
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


class UnsupportedMethod(Exception):
    """A known architecture restriction, rather than a fabricated attribution."""


@contextmanager
def seeded(torch, seed):
    state = np.random.get_state()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        np.random.seed(seed)
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            torch.backends.mha.set_fastpath_enabled(False)
            yield
    finally:
        np.random.set_state(state)
        torch.backends.mha.set_fastpath_enabled(fastpath)


def best_iou(heatmap, ground_truth):
    """Best threshold IoU over distinct saliency values; ties remain together.

    This is a GT-selected diagnostic, not an unbiased localization estimate.
    Ignored GT pixels are excluded from both intersection and union.
    """
    if ground_truth is None or not np.any(ground_truth['mask']):
        return None
    labels = ground_truth.get('label_map')
    valid = np.ones(heatmap.shape, dtype=bool) if labels is None else np.asarray(labels) != 1000
    scores = np.asarray(heatmap)[valid].ravel()
    truth = np.asarray(ground_truth['mask'], dtype=bool)[valid].ravel()
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_truth = scores[order], truth[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    true_positive = np.cumsum(sorted_truth)[ends]
    union = int(truth.sum()) + ends + 1 - true_positive
    ious = true_positive / np.maximum(union, 1)
    position = int(np.argmax(ious))
    threshold = float(sorted_scores[ends[position]])
    return {'best_iou': float(ious[position]), 'threshold': threshold,
            'threshold_selection': 'maximum IoU using GT on this image',
            'mask': (np.asarray(heatmap) >= threshold) & valid}


def lime_map(adapter, image, replacements, target, budget, args):
    from lime.lime_image import LimeImageExplainer
    from skimage.segmentation import slic
    # Use approximately budget/5 features, as in the original experiment.
    segments = slic(image, n_segments=max(2, budget // 5), compactness=10,
                    start_label=0, channel_axis=-1)
    result = []
    for background in replacements:
        exp = LimeImageExplainer(random_state=args.explanation_seed).explain_instance(
            image, adapter.predict_batch, labels=(target,), top_labels=None,
            hide_color=background, num_features=len(np.unique(segments)), num_samples=budget,
            batch_size=args.model_batch_size, segmentation_fn=lambda _: segments,
            random_seed=args.explanation_seed)
        weights = dict(exp.local_exp[target])
        result.append(np.vectorize(lambda label: weights.get(int(label), 0.0))(segments))
    return np.mean(result, axis=0)[None], {'samples_per_background': budget,
                                         'background_count': len(replacements),
                                         'regions': len(np.unique(segments)), 'segmentation': 'SLIC',
                                         'local_surrogate_r2': float(exp.score)}


def _residual_forward(self, x):
    identity = x
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.relu2(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    return self.relu3(self.addition(out, identity))


def lrp_model(model):
    """Independent residual nodes for Captum; preserve the scoring model."""
    import torch
    from torchvision.models.resnet import Bottleneck
    from captum.attr._utils.custom_modules import Addition_Module
    result = copy.deepcopy(model)
    for module in result.modules():
        if isinstance(module, Bottleneck):
            module.relu1 = torch.nn.ReLU()
            module.relu2 = torch.nn.ReLU()
            module.relu3 = torch.nn.ReLU()
            module.addition = Addition_Module()
            # Remove the unused shared ReLU so Captum only hooks executed nodes.
            del module.relu
            module.forward = MethodType(_residual_forward, module)
        elif isinstance(module, torch.nn.ReLU):
            module.inplace = False
    return result.eval()


class GradientMethods:
    def __init__(self, adapter, image, replacements, args):
        import torch
        from PIL import Image
        if not hasattr(adapter, 'preprocess') or not hasattr(adapter, 'model'):
            raise UnsupportedMethod('Gradient baselines require a torchvision image classifier; detector scoring has no gradient adapter')
        self.adapter, self.args, self.torch = adapter, args, torch
        self.image = image
        self.model = adapter.model
        self.device = adapter.device
        self.x = adapter.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        self.baselines = torch.stack([adapter.preprocess(Image.fromarray(background))
                                     for background in replacements]).to(self.device)
        self.forward = lambda x: torch.softmax(self.model(x), dim=1)

    def restore(self, values):
        """Undo center crop/resize; assign zero outside the model's field of view."""
        import torchvision.transforms.functional as F
        torch = self.torch
        values = values.detach().float()
        while values.ndim > 2:
            values = values.sum(dim=0)
        crop_h, crop_w = self.x.shape[-2:]
        values = torch.nn.functional.interpolate(values[None, None], (crop_h, crop_w),
                                                  mode='bilinear', align_corners=False)[0, 0]
        raw = torch.from_numpy(np.array(self.image)).permute(2, 0, 1).float()
        resized = F.resize(raw, self.adapter.preprocess.resize_size,
                           interpolation=self.adapter.preprocess.interpolation,
                           antialias=self.adapter.preprocess.antialias)
        rh, rw = resized.shape[-2:]
        # Torchvision center_crop rounds the half-offset, including odd sizes.
        padded_h, padded_w = max(rh, crop_h), max(rw, crop_w)
        canvas = torch.zeros((padded_h, padded_w), device=values.device)
        top, left = int(round((padded_h-crop_h)/2)), int(round((padded_w-crop_w)/2))
        canvas[top:top+crop_h, left:left+crop_w] = values
        pad_top, pad_left = (padded_h-rh)//2, (padded_w-rw)//2
        canvas = canvas[pad_top:pad_top+rh, pad_left:pad_left+rw]
        output = torch.nn.functional.interpolate(canvas[None, None], self.image.shape[:2],
                                                 mode='bilinear', align_corners=False)
        return output[0].cpu().numpy()

    def compute(self, method, target):
        torch, args = self.torch, self.args
        x = self.x.detach().requires_grad_(True)
        with torch.enable_grad(), seeded(torch, args.explanation_seed):
            if method == 'LRP':
                if getattr(self.adapter, 'name', '') != 'resnet50':
                    raise UnsupportedMethod('LRP for torchvision ViT needs a transformer-specific relevance backend; no substitute map is generated')
                from captum.attr import LRP
                model = lrp_model(self.model)
                try:
                    with torch.no_grad():
                        if not torch.allclose(model(x), self.model(x), atol=1e-5, rtol=1e-4):
                            raise ValueError('LRP residual conversion changed model scores')
                    values = LRP(model).attribute(x, target=target)
                finally:
                    del model
            elif method == 'GradCAM':
                from captum.attr import LayerGradCam
                if getattr(self.adapter, 'name', '') == 'resnet50':
                    layer = self.model.layer4[-1]
                    values = LayerGradCam(self.forward, layer).attribute(x, target=target, relu_attributions=True)
                elif getattr(self.adapter, 'name', '') == 'vit_b_16':
                    # Patch tokens before the last attention block still influence CLS.
                    layer = self.model.encoder.layers[-1].ln_1
                    saved = []
                    def capture(_, __, output):
                        saved.append(output)
                    hook = layer.register_forward_hook(capture)
                    try:
                        score = self.forward(x)[:, target].sum()
                        features = saved[-1]
                        gradient = torch.autograd.grad(score, features)[0]
                        side = int((features.shape[1]-1) ** .5)
                        weights = gradient[:, 1:].mean(dim=1, keepdim=True)
                        values = (features[:, 1:] * weights).sum(dim=-1).relu().reshape(1, 1, side, side)
                    finally:
                        hook.remove()
                else:
                    raise UnsupportedMethod('No GradCAM target layer configured for this architecture')
            elif method == 'GradExp':
                import shap
                explainer = shap.GradientExplainer(self.forward_module(target), self.baselines,
                                                   batch_size=args.attribution_batch_size)
                values = explainer.shap_values(x, nsamples=args.gradient_samples,
                                                ranked_outputs=None, rseed=args.explanation_seed)
                if isinstance(values, list):
                    values = values[0]
                else:
                    values = np.asarray(values)[..., 0]
                values = torch.as_tensor(values, device=self.device)
            elif method == 'GradShap':
                from captum.attr import GradientShap
                # Bound memory by accumulating sample-weighted chunks.
                values = torch.zeros_like(x)
                remaining = args.gradient_samples
                while remaining:
                    count = min(remaining, args.attribution_batch_size)
                    values += GradientShap(self.forward).attribute(
                        x, baselines=self.baselines, target=target, n_samples=count, stdevs=.0001) * count
                    remaining -= count
                values /= args.gradient_samples
            elif method == 'IDG':
                values = self.idg(x, target)
            else:
                raise ValueError(method)
        self.model.zero_grad(set_to_none=True)
        return self.restore(values)

    def forward_module(self, target):
        torch, model = self.torch, self.model
        class ProbabilityModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
            def forward(self, x):
                return torch.softmax(self.model(x), dim=1)[:, target:target+1]
        return ProbabilityModel().eval()

    def idg(self, x, target):
        """Adaptive, decision-slope-weighted path gradients (IDG).

        Uses the published decision-slope sampling strategy, with guards for
        constant scores and partial batches. Empty intervals keep their position.
        """
        torch, args = self.torch, self.args
        steps = args.idg_steps
        total = torch.zeros_like(x)
        for baseline in self.baselines:
            baseline = baseline[None]
            difference = x.detach() - baseline
            grid = torch.linspace(0, 1, steps, device=x.device)
            with torch.no_grad():
                scores = torch.cat([self.forward(baseline + grid[i:i+args.attribution_batch_size, None, None, None] * difference)[:, target]
                                    for i in range(0, steps, args.attribution_batch_size)])
            slopes = torch.cat([scores.new_zeros(1), scores.diff() * (steps-1)])
            allocation = slopes - slopes.min()
            allocation[0] = 0
            if float(allocation.sum()) < 1e-12:
                allocation = torch.ones_like(allocation)
                allocation[0] = 0
            expected = allocation / allocation.sum() * steps
            counts = expected.floor().long()
            remainder = steps - int(counts.sum())
            counts[torch.argsort(expected-counts, descending=True)[:remainder]] += 1
            alphas, widths = [], []
            for index in range(1, steps):
                count = int(counts[index])
                if count:
                    alphas.append((index-1 + torch.arange(count, device=x.device)/count)/(steps-1))
                    widths.append(torch.full((count,), 1/((steps-1)*count), device=x.device))
            alphas, widths = torch.cat(alphas), torch.cat(widths)
            gradients, decisions = [], []
            for start in range(0, len(alphas), args.attribution_batch_size):
                points = (baseline + alphas[start:start+args.attribution_batch_size, None, None, None] * difference).detach().requires_grad_(True)
                score = self.forward(points)[:, target]
                gradients.append(torch.autograd.grad(score.sum(), points)[0].detach())
                decisions.append(score.detach())
            decisions = torch.cat(decisions)
            local_slopes = torch.cat([decisions.new_zeros(1), decisions.diff()/alphas.diff().clamp_min(1e-12)])
            weighted = torch.cat(gradients) * (local_slopes*widths)[:, None, None, None]
            total += weighted.mean(dim=0, keepdim=True) * difference
        return total / len(self.baselines)


def compute_extra(runtime, args, adapter, image, replacements, explainer, tree, target):
    """Yield name, map, status, seconds for every requested baseline."""
    gradients = None
    gradient_error = None
    for name in method_names(args):
        started = time.perf_counter()
        info = {'status': 'complete'}
        try:
            if name.startswith(('BPT-', 'AA-')):
                method, budget = name.split('-')
                values = explainer.explain_instance(int(budget), bpt=tree if method == 'BPT' else None,
                                                    method=method, batch_size=args.eval_batch_size)
                info['budget'] = int(budget)
            elif name.startswith('LIME-'):
                values, details = lime_map(adapter, image, replacements, target, int(name.split('-')[1]), args)
                info.update(details)
            else:
                if gradient_error:
                    raise UnsupportedMethod(gradient_error)
                if gradients is None:
                    try:
                        gradients = GradientMethods(adapter, image, replacements, args)
                    except UnsupportedMethod as error:
                        gradient_error = str(error)
                        raise
                values = gradients.compute(name, target)
                info.update(score='logit' if name == 'LRP' else 'classification probability',
                            reduction='signed sum over input channels',
                            alignment='inverse model resize and center crop; zero outside crop')
            values = np.asarray(values, dtype=float)
            if values.shape != (1, *image.shape[:2]) or not np.isfinite(values).all():
                raise ValueError(f'{name}: invalid attribution map {values.shape}')
        except UnsupportedMethod as error:
            values = None
            info = {'status': 'unsupported', 'reason': str(error)}
        yield name, values, info, time.perf_counter()-started


def comparison_plot(runtime, image, attributions, prediction, output, ground_truth, statuses):
    """Scrollable heatmaps and oracle-threshold GT overlap, like the notebook."""
    plt = runtime.plt
    names = list(statuses)
    columns = 1 + int(ground_truth is not None) + len(names)
    has_gt = ground_truth is not None and np.any(ground_truth['mask'])
    rows = 2 if has_gt else 1
    fig, axes = plt.subplots(rows, columns, figsize=(2.1*columns, 2.5*rows), squeeze=False)
    iou_results = {}
    try:
        for row in axes:
            row[0].imshow(image)
            row[0].set_title('Input')
            if ground_truth is not None:
                row[1].imshow(ground_truth['mask'], cmap='gray', vmin=0, vmax=1)
                row[1].set_title('GT' if has_gt else f"GT: {ground_truth['status']}", fontsize=8)
        offset = 1 + int(ground_truth is not None)
        for column, name in enumerate(names, offset):
            heatmap = attributions.get(name)
            ax = axes[0, column]
            ax.set_title(name, fontsize=10)
            if heatmap is None:
                for row in axes:
                    row[column].text(.5, .5, statuses[name]['status'] + '\n' + textwrap.fill(statuses[name].get('reason', ''), 25),
                                     ha='center', va='center', wrap=True, fontsize=7,
                                     transform=row[column].transAxes)
                continue
            heatmap = np.asarray(heatmap)[0]
            limit = max(float(np.nanpercentile(np.abs(heatmap), 99.5)), 1e-12)
            ax.imshow(heatmap, cmap='coolwarm', vmin=-limit, vmax=limit)
            if has_gt:
                mask = np.asarray(ground_truth['mask'], dtype=bool)
                ax.contour(np.arange(-1, mask.shape[1] + 1),
                           np.arange(-1, mask.shape[0] + 1),
                           np.pad(mask.astype(float), 1), levels=[.5],
                           colors='black', linewidths=.7)
                ax.set_xlim(-.5, mask.shape[1]-.5)
                ax.set_ylim(mask.shape[0]-.5, -.5)
                result = best_iou(heatmap, ground_truth)
                selected = result.pop('mask')
                iou_results[name] = result
                # TP blue, FP orange, FN magenta, TN white.
                overlap = np.ones((*mask.shape, 3))
                overlap[selected & mask] = [0, .15, 1]
                overlap[selected & ~mask] = [1, .55, 0]
                overlap[~selected & mask] = [1, 0, 1]
                axes[1, column].imshow(overlap)
                axes[1, column].set_title(f"Best IoU: {result['best_iou']:.3f}", fontsize=9)
        for ax in axes.ravel():
            ax.axis('off')
        top = (prediction.get('top_k_classes') or [{}])[0]
        top_text = top.get('class_name', 'unavailable')
        if 'confidence' in top:
            top_text += f" ({float(top['confidence']):.4f})"
        fig.suptitle(f"Top predicted: {top_text} | Explained: {prediction['explained_class']}" +
                     (' | IoU thresholds selected using GT' if has_gt else ''), fontsize=11)
        fig.subplots_adjust(left=.008, right=.992, bottom=.04, top=.85, wspace=.12, hspace=.35)
        fig.savefig(output, dpi=130)
    finally:
        plt.close(fig)
    return iou_results
