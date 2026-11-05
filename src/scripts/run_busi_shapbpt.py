#!/usr/bin/env python3
"""Frozen BUSI classifier + SAM region BPT + hierarchical permutation attribution.
Run from the repository root. No classifier training is performed.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


CLASSES = ("benign", "malignant", "normal")
DATASET = "aryashah2k/breast-ultrasound-images-dataset"
MODEL = "ghostsas001/busi-breast-ultrasound-siglip"


def discover_cases(root: Path):
    """Discover images and union matching *_mask*.png annotations."""
    cases = []
    for p in sorted(root.rglob("*.png")):
        if "_mask" in p.stem.lower():
            continue
        cls = next((part.lower() for part in p.parts if part.lower() in CLASSES), None)
        if cls is None:
            continue
        masks = sorted(p.parent.glob(p.stem + "_mask*.png"))
        cases.append({"image": str(p), "class": cls, "masks": [str(m) for m in masks]})
    return cases


def download_dataset(data_root: Path):
    """Download/validate BUSI with the standard library; no notebook dependencies."""
    import urllib.request
    import zipfile
    data_root.mkdir(parents=True, exist_ok=True)
    if discover_cases(data_root):
        print("[dataset] existing images found; skipping dataset download", flush=True)
        return
    archive = data_root / "BUSI_kaggle.zip"
    if not archive.is_file():
        temporary = archive.with_suffix(".zip.part")
        print("[download] BUSI Kaggle mirror", flush=True)
        urllib.request.urlretrieve(
            f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}", temporary)
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError(
                f"Download is not a ZIP: {temporary}. Kaggle may require browser login. "
                "Download the archive from the Kaggle website, extract it under --dataset, and retry.")
        temporary.replace(archive)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Invalid ZIP: {archive}; replace it with a valid BUSI archive")
    with zipfile.ZipFile(archive) as zipped:
        damaged = zipped.testzip()
        if damaged:
            raise RuntimeError(f"Damaged ZIP member: {damaged}; download the archive again")
        for member in zipped.infolist():
            destination = (data_root / member.filename).resolve()
            if not destination.is_relative_to(data_root.resolve()):
                raise RuntimeError(f"Unsafe ZIP path: {member.filename}")
        zipped.extractall(data_root)


def download_assets(data_root: Path, model_root: Path):
    download_dataset(data_root)
    cases = discover_cases(data_root)
    if not cases:
        raise RuntimeError("No BUSI images found. Check dataset layout.")
    # Set before importing the Hub: it reads transfer settings at import time.
    # EPITO login nodes may not support the environment's compiled Xet extension.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    model_root.mkdir(parents=True, exist_ok=True)
    print("[download] pretrained SigLIP checkpoint", flush=True)
    snapshot_download(MODEL, local_dir=str(model_root), allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"])
    print("[model]", model_root)
    print("[cases]", len(cases), "by class", {c: sum(x["class"] == c for x in cases) for c in CLASSES})
    return cases


def load_classifier(model_root: Path, device: str):
    import torch
    from transformers import AutoModelForImageClassification
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor
    model = AutoModelForImageClassification.from_pretrained(str(model_root), use_safetensors=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    # Explicitly match the model card, not default AutoProcessor transforms.
    preprocessing = Compose([Resize((224, 224)), ToTensor(), Normalize([0.5] * 3, [0.5] * 3)])
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    return model, preprocessing, labels


def score_images(images, model, preprocessing, device, target_idx, batch_size):
    """Return target-class logits and probabilities for PIL RGB images."""
    import numpy as np
    import torch
    logits_out, probs_out = [], []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.stack([preprocessing(im.convert("RGB")) for im in images[start:start + batch_size]]).to(device)
            z = model(pixel_values=x).logits
            logits_out.extend(z[:, target_idx].float().cpu().numpy().tolist())
            probs_out.extend(torch.softmax(z.float(), dim=1)[:, target_idx].cpu().numpy().tolist())
    return np.asarray(logits_out, dtype=np.float64), np.asarray(probs_out, dtype=np.float64)


def load_and_validate_tree(path: Path, labels: np.ndarray):
    import numpy as np
    tree = json.loads(path.read_text())
    nodes = {int(n["id"]): n for n in tree["nodes"]}
    root = int(tree["root"])
    assert len(nodes) == len(tree["nodes"]) and root in nodes
    leaves = {int(n["id"]): int(n["original_label"]) for n in nodes.values() if not n["children"]}
    assert set(leaves.values()) == set(int(v) for v in np.unique(labels)), "Leaves and labels mismatch"
    parents = {k: 0 for k in nodes}
    for n in nodes.values():
        assert len(n["children"]) in (0, 2)
        for c in n["children"]:
            parents[int(c)] += 1
    assert parents[root] == 0 and all(v == 1 for k, v in parents.items() if k != root)
    visited = set()
    def visit(k):
        assert k not in visited, "Tree contains cycle"
        visited.add(k)
        for c in nodes[k]["children"]:
            visit(int(c))
    visit(root)
    assert len(visited) == len(nodes)
    return nodes, root, leaves


def random_hierarchical_order(nodes, root, rng):
    """Uniformly shuffle siblings at each binary node, preserving subtrees."""
    def walk(k):
        children = nodes[k]["children"]
        if not children:
            return [int(k)]
        a, b = map(int, children)
        if rng.integers(2):
            a, b = b, a
        return walk(a) + walk(b)
    return walk(root)


def mask_coalition(image: Image.Image, labels: np.ndarray, retained: set[int], baseline: Image.Image):
    import numpy as np
    from PIL import Image, ImageFilter
    orig = np.asarray(image.convert("RGB"))
    bg = np.asarray(baseline.convert("RGB"))
    keep = np.isin(labels, list(retained))
    return Image.fromarray(np.where(keep[..., None], orig, bg).astype(np.uint8), "RGB")


def estimate_owen(image, labels, nodes, root, leaves, model, prep, device, target, batch, budget, baseline, seed):
    """Hierarchical random-permutation marginal estimator. Budget counts model evaluations."""
    import numpy as np
    n = len(leaves)
    samples = max(1, budget // (n + 1))
    rng = np.random.default_rng(seed)
    phi = {k: 0.0 for k in leaves}
    start = time.perf_counter()
    for rep in range(samples):
        order = random_hierarchical_order(nodes, root, rng)
        retained = set()
        coalitions = [mask_coalition(image, labels, retained, baseline)]
        for leaf in order:
            retained.add(leaves[leaf])
            coalitions.append(mask_coalition(image, labels, retained, baseline))
        scores, _ = score_images(coalitions, model, prep, device, target, batch)
        for i, leaf in enumerate(order):
            phi[leaf] += float(scores[i+1] - scores[i]) / samples
        print(f"[owen] permutation {rep+1}/{samples} | evaluations={(rep+1)*(n+1)}", flush=True)
    heat = np.zeros(labels.shape, dtype=np.float32)
    for leaf, lab in leaves.items():
        heat[labels == lab] = phi[leaf]
    return heat, phi, {"model_evaluations": samples * (n + 1), "samples": samples, "seconds": time.perf_counter()-start}


def evaluate_curves(image, labels, heat, model, prep, device, target, batch, baseline, steps=21):
    """Insertion/deletion of whole leaves ranked by attribution, scored by probability."""
    import numpy as np
    unique = np.unique(labels)
    strengths = [(float(heat[labels == lab].mean()), int(lab)) for lab in unique]
    order = [lab for _, lab in sorted(strengths, reverse=True)]
    curves = {}
    for mode in ("insertion", "deletion"):
        images = []
        for k in np.linspace(0, len(order), steps).round().astype(int):
            present = set(order[:k]) if mode == "insertion" else set(order[k:])
            images.append(mask_coalition(image, labels, present, baseline))
        _, probs = score_images(images, model, prep, device, target, batch)
        curves[mode] = {"probabilities": probs.tolist(), "auc": float(np.trapz(probs, dx=1/(steps-1)))}
    return curves


def union_ground_truth(mask_paths, shape):
    import numpy as np
    from PIL import Image, ImageFilter
    if not mask_paths:
        return None
    gt = np.zeros(shape, dtype=bool)
    for p in mask_paths:
        mask = Image.open(p).convert("L")
        if mask.size != (shape[1], shape[0]):
            mask = mask.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
        gt |= (np.asarray(mask) > 127)
    return gt


def localization(heat, gt):
    import numpy as np
    if gt is None or not gt.any():
        return {"pointing": None, "top_area_iou": None}
    # Positive evidence only. The top |GT| pixels are selected for size-matched IoU.
    evidence = np.maximum(heat, 0.0)
    point = np.unravel_index(np.argmax(evidence), evidence.shape)
    k = int(gt.sum())
    flat = evidence.ravel()
    selected = np.zeros(flat.size, dtype=bool)
    selected[np.argpartition(flat, -k)[-k:]] = True
    selected = selected.reshape(gt.shape)
    inter = np.logical_and(gt, selected).sum()
    union = np.logical_or(gt, selected).sum()
    return {"pointing": bool(gt[point]), "top_area_iou": float(inter / union), "lesion_pixels": k}


def save_fig(image, labels, heat, gt, path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    img = np.asarray(image)
    overlay = img.copy()
    overlay[boundary] = (255, 60, 60)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    axes[0].imshow(img); axes[0].set_title("BUSI input")
    axes[1].imshow(overlay); axes[1].set_title("SAM / BPT leaves")
    axes[2].imshow(img, cmap="gray"); axes[2].imshow(heat, cmap="coolwarm", alpha=.6)
    axes[2].set_title("Hierarchical marginal attribution")
    axes[3].imshow(img, cmap="gray")
    if gt is not None:
        outline = binary_dilation(gt) & ~gt
        mask_overlay = np.zeros((*gt.shape, 4), dtype=float)
        mask_overlay[outline] = [0, 1, .2, 1]
        axes[3].imshow(mask_overlay)
    axes[3].set_title("Reference lesion boundary")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)


def case_folder(case, args):
    name = f"{case['class']}_{Path(case['image']).stem.replace(' ', '_')}"
    return args.results / name


def case_signature(case, args):
    import hashlib
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key not in {'command', 'limit', 'results', 'dataset', 'resume', 'explain_all', 'config',
                               'partition_folder', 'show_gt', 'no_heatmaps', 'no_auc', 'auc_num_samples', 'bg_type'}}
    paths = [Path(case['image']), *map(Path, case['masks']),
             *args.model_dir.glob('*.json'), *args.model_dir.glob('*.safetensors')]
    paths += [getattr(args, method + '_checkpoint') for method in args.sam]
    stamps = [[str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]
              for path in sorted(set(paths))]
    return hashlib.sha256(json.dumps({'settings': settings, 'files': stamps}, sort_keys=True).encode()).hexdigest()


def case_artifacts(args):
    artifacts = ['input.png', 'metrics.json']
    for method in args.sam:
        artifacts.extend(f'{method}/{name}' for name in [
            f'input_{method}_refined_{args.regions}.npy', 'tree.json', 'attribution.npy', 'explanation.png'])
    return artifacts


def completed_case(case, args):
    if not args.resume:
        return None
    folder = case_folder(case, args)
    marker = folder / 'complete.json'
    if not marker.is_file():
        return None
    try:
        metadata = json.loads(marker.read_text())
        if metadata['signature'] != case_signature(case, args):
            raise ValueError('BUSI settings/inputs changed; choose a new --results directory or use --no-resume')
        if metadata['artifacts'] != case_artifacts(args):
            return None
        if not all((folder / name).is_file() for name in metadata['artifacts']):
            return None
        return json.loads((folder / 'metrics.json').read_text())
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def write_case_completion(case, args, signature):
    folder = case_folder(case, args)
    artifacts = case_artifacts(args)
    if not all((folder / name).is_file() for name in artifacts):
        raise RuntimeError('Cannot complete BUSI image: output files are missing')
    temporary = folder / 'complete.json.tmp'
    temporary.write_text(json.dumps({'signature': signature, 'artifacts': artifacts}, indent=2))
    temporary.replace(folder / 'complete.json')


def process_one(case, args, model, prep, id2label):
    import numpy as np
    from PIL import Image, ImageFilter
    image_path = Path(case["image"])
    image = Image.open(image_path).convert("L").convert("RGB")
    # Unique stem avoids benign (1) / malignant (1) collisions.
    name = f"{case['class']}_{image_path.stem.replace(' ', '_')}"
    out = args.results / name
    out.mkdir(parents=True, exist_ok=True)
    image_file = out / "input.png"
    image.save(image_file)
    logits = model_predict_all(image, model, prep, args.device)
    pred = int(logits.argmax())
    label = id2label[pred]
    print(f"[case] {image_path.name} GT={case['class']} predicted={label}", flush=True)
    gt = union_ground_truth(case["masks"], (image.height, image.width))
    baseline = Image.new("RGB", image.size, color=(127, 127, 127)) if args.baseline == "gray" else image.filter(ImageFilter.GaussianBlur(radius=16))
    metrics = {"image": str(image_path), "ground_truth_class": case["class"], "prediction": label, "target_index": pred, "probabilities": model_softmax(logits).tolist(), "methods": {}}
    for method in args.sam:
        directory = out / method
        directory.mkdir(exist_ok=True)
        labels_file = directory / f"input_{method}_refined_{args.regions}.npy"
        tree_file = directory / "tree.json"
        if not labels_file.exists():
            cmd = [sys.executable, "src/scripts/generate_sam_partitions.py", "--model", method,
                   "--image", str(image_file), "--device", args.device, "--max-regions", str(args.regions),
                   "--strategy", "refined", "--output-dir", str(directory),
                   "--points-per-side", str(args.points_per_side), "--points-per-batch", str(args.points_per_batch)]
            if method == "sam1":
                cmd += ["--checkpoint", str(args.sam1_checkpoint)]
            else:
                cmd += ["--checkpoint", str(args.sam2_checkpoint), "--sam2-config", args.sam2_config]
            subprocess.run(cmd, check=True)
        labels = np.load(labels_file)
        if not tree_file.exists():
            subprocess.run([sys.executable, "src/scripts/build_sam_bpt.py", "--image", str(image_file),
                            "--labels", str(labels_file), "--out", str(tree_file)], check=True)
        nodes, root, leaves = load_and_validate_tree(tree_file, labels)
        heat, phi, cost = estimate_owen(image, labels, nodes, root, leaves, model, prep, args.device,
                                        pred, args.batch_size, args.budget, baseline, args.seed)
        curves = evaluate_curves(image, labels, heat, model, prep, args.device, pred, args.batch_size, baseline)
        loc = localization(heat, gt)
        np.save(directory / "attribution.npy", heat)
        save_fig(image, labels, heat, gt, directory / "explanation.png")
        metrics["methods"][method] = {"tree_leaves": len(leaves), "cost": cost, "curves": curves,
                                     "localization": loc, "attributions_by_leaf": {str(k): v for k,v in phi.items()}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def model_predict_all(image, model, prep, device):
    import torch
    with torch.inference_mode():
        return model(pixel_values=prep(image).unsqueeze(0).to(device)).logits[0].float().cpu().numpy()


def model_softmax(z):
    import numpy as np
    z = np.asarray(z, dtype=np.float64); q = np.exp(z-z.max()); return q/q.sum()


def run_all_methods(args):
    """Use the shared XAI runner on previously generated compact BUSI partitions."""
    import copy
    import yaml
    root = Path(__file__).resolve().parents[2]
    for directory in [root, root / 'src/scripts', root / 'src/scripts/explanations']:
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from config_paths import load_path_config, output_path
    from experiment_common import create_parser, run
    config = load_path_config(args.config)
    base = output_path(config, 'partitions')
    if args.partition_folder and len(args.sam) != 1:
        raise ValueError('--partition-folder requires one --sam model')
    results = []
    for sam_model in args.sam:
        selected = copy.deepcopy(config)
        configured = base / config['data']['mask_dir_final']
        prefix = f"BUSI_{sam_model}_"
        def matching(path):
            return (path.is_dir() and path.name.startswith(prefix)
                    and f'_{args.regions}_' in path.name and 'compact' in path.name)
        if args.partition_folder:
            folder = Path(args.partition_folder)
            folder = folder if folder.is_absolute() else base / folder
            if not matching(folder):
                raise ValueError(f'Expected a {sam_model} compact partition folder for {args.regions} regions: {folder}')
        elif matching(configured):
            folder = configured
        else:
            matches = sorted(path for path in base.glob(prefix + '*') if matching(path))
            if len(matches) != 1:
                raise ValueError(f'Expected one saved compact {sam_model} partition folder; found {len(matches)}. '
                                 'Generate BUSI partitions first or pass --partition-folder FOLDER_NAME. '
                                 f'Candidates: {[path.name for path in matches]}')
            folder = matches[0]
        selected['data']['mask_dir_final'] = str(folder.resolve())
        directory = args.results / 'all_methods_configs'
        directory.mkdir(parents=True, exist_ok=True)
        path = (directory / f'{folder.name}.yaml').resolve()
        content = yaml.safe_dump(selected, sort_keys=False)
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
        command = ['--config', str(path), '--checkpoint', str(args.model_dir),
                   '--limit', str(args.limit), '--budget', str(args.budget), '--explain_all',
                   '--device', args.device, '--bg-type', args.bg_type,
                   '--model-batch-size', str(args.batch_size), '--eval-batch-size', str(args.batch_size),
                   '--auc-num-samples', str(args.auc_num_samples), '--explanation-seed', str(args.seed),
                   '--local-files-only']
        for flag, enabled in [('--show-gt', args.show_gt), ('--no-heatmaps', args.no_heatmaps),
                              ('--no-resume', not args.resume), ('--no-auc', args.no_auc)]:
            if enabled:
                command.append(flag)
        print(f'[all methods] {sam_model}: saved compact partitions {folder}', flush=True)
        results.append(run('OC_BUSI', create_parser('OC_BUSI').parse_args(command)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "run", "all", "inventory"])
    parser.add_argument("--dataset", type=Path, default=Path("/beegfs/home/mrashid/datasets/BUSI"))
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/busi_siglip"))
    parser.add_argument("--results", type=Path, default=Path("results/busi"))
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Recompute XAI; existing SAM partitions/trees are still reused")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--sam", nargs="+", choices=["sam1", "sam2"], default=["sam1", "sam2"])
    parser.add_argument("--regions", type=int, default=62)
    parser.add_argument("--budget", type=int, default=504, help="minimum budget; rounded down to complete permutations")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", choices=["gray", "blur"], default="gray")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--sam1-checkpoint", type=Path, default=Path("checkpoints/sam1/sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("checkpoints/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--points-per-side", type=int, default=24)
    parser.add_argument("--points-per-batch", type=int, default=32)
    parser.add_argument('--explain_all', '--explain-all', action='store_true',
                        help='Run the shared OC_BUSI comparison on saved compact partitions')
    parser.add_argument('--config', default='BUSI_epito', help='Dataset/output config for --explain_all')
    parser.add_argument('--partition-folder', help='Saved partition folder name or absolute path for one --sam model')
    parser.add_argument('--show-gt', '--show_gt', action='store_true')
    parser.add_argument('--no-heatmaps', action='store_true')
    parser.add_argument('--no-auc', action='store_true')
    parser.add_argument('--auc-num-samples', type=int, default=200)
    parser.add_argument('--bg-type', choices=['black', 'gray', 'white', 'blurred', 'noise', 'full'], default='black')
    args = parser.parse_args()
    if args.command in ("download", "all"):
        download_assets(args.dataset, args.model_dir)
        if args.command == "download": return
    if args.explain_all:
        if args.command not in {'run', 'all'}:
            parser.error('--explain_all applies to run/all commands')
        return run_all_methods(args)
    cases = discover_cases(args.dataset)
    if not cases:
        raise RuntimeError(f"No BUSI images found in {args.dataset}. Run download first.")
    print("[inventory]", len(cases), "cases; class distribution:", {c: sum(x["class"] == c for x in cases) for c in CLASSES})
    if args.command == "inventory": return
    if not args.model_dir.is_dir() or not list(args.model_dir.glob("*.safetensors")):
        raise RuntimeError("Pretrained checkpoint missing. Run download first.")
    if args.limit < 1: raise ValueError("--limit must be >= 1")
    if args.regions < 1: raise ValueError("--regions must be >= 1")
    args.results.mkdir(parents=True, exist_ok=True)
    # Sort by class for a mixed-class smoke test (unlike default path order).
    chosen = []
    buckets = {c: [x for x in cases if x["class"] == c] for c in CLASSES}
    while len(chosen) < min(args.limit, len(cases)):
        for c in CLASSES:
            if buckets[c] and len(chosen) < args.limit:
                chosen.append(buckets[c].pop(0))
    all_metrics, pending = [], []
    for case in chosen:
        previous = completed_case(case, args)
        if previous is None:
            pending.append(case)
        else:
            print(f"[resume] skipping completed image: {case['image']}", flush=True)
            all_metrics.append(previous)
    print(f'[resume] complete={len(all_metrics)}; pending={len(pending)}', flush=True)
    if pending:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: run inside an allocated GPU compute node")
        model, prep, labels = load_classifier(args.model_dir, args.device)
        for case in pending:
            signature = case_signature(case, args)
            (case_folder(case, args) / 'complete.json').unlink(missing_ok=True)
            metrics = process_one(case, args, model, prep, labels)
            write_case_completion(case, args, signature)
            all_metrics.append(metrics)
            temporary = args.results / 'summary.json.tmp'
            temporary.write_text(json.dumps(all_metrics, indent=2))
            temporary.replace(args.results / 'summary.json')
    (args.results / "summary.json").write_text(json.dumps(all_metrics, indent=2))
    print("[done]", args.results / "summary.json")

if __name__ == "__main__":
    main()
