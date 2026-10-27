#!/usr/bin/env python3
"""Generate SAM1/SAM2 automatic masks and disjoint ShapBPT-ready region maps.

Dependencies: numpy, scipy, pillow, matplotlib, torch, and SAM1/SAM2 as needed.
Runs from any working directory; use --raw-masks to refine saved SAM masks offline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["sam1", "sam2"], default="sam2")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--max-regions", type=int, default=62)
    p.add_argument("--strategy", choices=["refined", "raw_overwrite"], default="refined")
    p.add_argument("--output-dir", type=Path, default=Path("results/partitions/sam2"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--sam2-config", default=None,
                   help="Hydra config name, e.g. configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--sam1-variant", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    p.add_argument("--points-per-side", type=int, default=16)
    p.add_argument("--points-per-batch", type=int, default=16)
    p.add_argument("--pred-iou-thresh", type=float, default=0.7)
    p.add_argument("--stability-score-thresh", type=float, default=0.85)
    p.add_argument("--min-region-pixels", type=int, default=64)
    p.add_argument("--raw-masks", type=Path, default=None,
                   help="Optional .npz containing 'masks' [N,H,W]; skip inference")
    return p.parse_args()


def locate_sam2_checkpoint():
    """Try paths used by this project and its third_party installations."""
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "checkpoints/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        root / "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        root.parent / "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        Path.cwd() / "checkpoints/sam2.1_hiera_tiny.pt",
    ]
    return next((f for f in candidates if f.is_file()), None)


def get_device(requested):
    import torch
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return requested


def infer_masks(image, args):
    import torch
    device = get_device(args.device)
    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.model == "sam2":
            checkpoint = locate_sam2_checkpoint()
        else:
            root = Path(__file__).resolve().parents[2]
            checkpoint = root / "checkpoints/sam1/sam_vit_b_01ec64.pth"
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {checkpoint}. Download the model or pass --checkpoint PATH"
        )
    print(f"[model] {args.model}  [device] {device}  [checkpoint] {checkpoint}", flush=True)
    if args.model == "sam2":
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        config = args.sam2_config or "configs/sam2.1/sam2.1_hiera_t.yaml"
        model = build_sam2(config, str(checkpoint), device=device, apply_postprocessing=False)
        generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            crop_n_layers=0,
            output_mode="binary_mask",
        )
    else:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        model = sam_model_registry[args.sam1_variant](checkpoint=str(checkpoint)).to(device)
        generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            crop_n_layers=0,
            output_mode="binary_mask",
        )
        # SAM1's automatic point grids default to float64 on some versions.
        # Apple MPS cannot create float64 tensors; use float32 coordinates.
        if device == "mps":
            generator.point_grids = [
                grid.astype(np.float32, copy=False)
                for grid in generator.point_grids
            ]
            # SAM1 ResizeLongestSide.apply_coords scales points using Python
            # floats and can promote float32 inputs back to float64. The
            # automatic-mask generator then creates an MPS tensor from those
            # transformed coordinates, which fails for float64.
            original_apply_coords = generator.predictor.transform.apply_coords

            def apply_coords_float32(coords, original_size):
                transformed = original_apply_coords(coords, original_size)
                return np.asarray(transformed, dtype=np.float32)

            generator.predictor.transform.apply_coords = apply_coords_float32
    t0 = time.perf_counter()
    with torch.inference_mode():
        annotations = generator.generate(image)
    dt = time.perf_counter() - t0
    masks = np.stack([m["segmentation"] for m in annotations]).astype(bool) if annotations else np.empty((0, *image.shape[:2]), dtype=bool)
    return masks, dt, device, str(checkpoint)


def consecutive_labels(arr):
    _, inverse = np.unique(arr, return_inverse=True)
    return inverse.reshape(arr.shape).astype(np.uint32)


def raw_overwrite(masks):
    h, w = masks.shape[1:]
    result = np.zeros((h, w), dtype=np.uint32)
    order = np.argsort(masks.sum(axis=(1, 2)))[::-1]
    for idx, k in enumerate(order, 1):
        result[masks[k]] = idx
    return result


def nearest_fill(labels):
    unassigned = labels == 0
    if not unassigned.any() or unassigned.all():
        return labels
    _, indices = ndi.distance_transform_edt(unassigned, return_indices=True)
    labels = labels.copy()
    labels[unassigned] = labels[tuple(axis[unassigned] for axis in indices)]
    return labels


def split_connected(labels, min_pixels):
    """Split disconnected components; discard very small components for nearest filling."""
    output = np.zeros(labels.shape, dtype=np.uint32)
    next_label = 0
    structure = ndi.generate_binary_structure(2, 1)
    for value in np.unique(labels):
        if value == 0:
            continue
        components, number = ndi.label(labels == value, structure=structure)
        if not number:
            continue
        counts = np.bincount(components.ravel())
        for component in range(1, number + 1):
            if counts[component] < min_pixels:
                continue
            next_label += 1
            output[components == component] = next_label
    return output


def enforce_region_budget(labels, max_regions):
    """Retain largest max_regions; assign removed pixels to nearest retained region."""
    counts = np.bincount(labels.ravel())
    present = np.flatnonzero(counts)
    present = present[present != 0]
    if len(present) <= max_regions:
        return consecutive_labels(labels)
    kept = present[np.argsort(counts[present], kind="stable")[-max_regions:]]
    keep_pixels = np.isin(labels, kept)
    if not keep_pixels.any():
        raise ValueError("No partition regions available")
    _, nearest = ndi.distance_transform_edt(~keep_pixels, return_indices=True)
    result = labels.copy()
    missing = ~keep_pixels
    result[missing] = labels[tuple(axis[missing] for axis in nearest)]
    return consecutive_labels(result)


def refined_partitions(masks, min_region_pixels, max_regions):
    h, w = masks.shape[1:]
    if masks.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint16)
    # Largest first, then smaller masks overwrite: preserves small object structure.
    label_map = raw_overwrite(masks)
    # Reserve zero only for uncovered pixels until they are attached to a region.
    label_map = nearest_fill(label_map)
    components = split_connected(label_map, min_region_pixels)
    if not np.any(components):
        # Very small image / restrictive min-region threshold.
        components = label_map
    components = nearest_fill(components)
    result = enforce_region_budget(components, max_regions)
    return result.astype(np.uint16 if max_regions <= 65535 else np.uint32)


def save_plot(image, partitions, target, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    boundary = find_boundaries(partitions, mode="thick")
    overlay = image.astype(np.float32).copy()
    overlay[boundary] = np.array([255, 30, 30], dtype=np.float32)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(image)
    axs[0].set_title("Input image")
    axs[1].imshow(partitions, cmap="nipy_spectral", interpolation="nearest")
    axs[1].set_title(f"{title}: {len(np.unique(partitions))} regions")
    axs[2].imshow(overlay.astype(np.uint8))
    axs[2].set_title("Partition boundaries")
    for ax in axs:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.max_regions < 1:
        raise ValueError("--max-regions must be >=1")
    image = np.asarray(Image.open(args.image).convert("RGB"))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    if args.raw_masks:
        with np.load(args.raw_masks, allow_pickle=False) as loaded:
            masks = loaded["masks"].astype(bool)
        inference_seconds, device, checkpoint = None, "cached", None
    else:
        masks, inference_seconds, device, checkpoint = infer_masks(image, args)
    if masks.ndim != 3 or masks.shape[1:] != image.shape[:2]:
        raise ValueError(f"Expected masks [N,{image.shape[0]},{image.shape[1]}], got {masks.shape}")
    masks_path = out / f"{stem}_{args.model}_raw_masks.npz"
    np.savez_compressed(masks_path, masks=masks)
    raw_union = np.any(masks, axis=0) if len(masks) else np.zeros(image.shape[:2], dtype=bool)
    t0 = time.perf_counter()
    if args.strategy == "refined":
        partitions = refined_partitions(masks, args.min_region_pixels, args.max_regions)
    else:
        partitions = raw_overwrite(masks)
        partitions = enforce_region_budget(partitions, args.max_regions).astype(np.uint16)
    refinement_seconds = time.perf_counter() - t0
    unique, counts = np.unique(partitions, return_counts=True)
    basename = f"{stem}_{args.model}_{args.strategy}_{args.max_regions}"
    np.save(out / f"{basename}.npy", partitions)
    save_plot(image, partitions, out / f"{basename}.png", args.strategy)
    metadata = {
        "image": str(args.image), "image_shape": list(image.shape),
        "model": args.model, "strategy": args.strategy, "device": device,
        "checkpoint": checkpoint, "max_regions": args.max_regions,
        "region_count": int(len(unique)), "raw_masks_count": int(len(masks)),
        "raw_union_coverage_percent": round(100 * float(raw_union.mean()), 4),
        "uncovered_pixels_before_filling": int((~raw_union).sum()),
        "final_assigned_pixels": int(partitions.size),
        "region_pixel_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "inference_seconds": inference_seconds, "refinement_seconds": round(refinement_seconds, 3),
        "config": {"points_per_side": args.points_per_side,
                   "points_per_batch": args.points_per_batch,
                   "pred_iou_thresh": args.pred_iou_thresh,
                   "stability_score_thresh": args.stability_score_thresh,
                   "min_region_pixels": args.min_region_pixels},
        "region_ids_are_labels_not_semantic_classes": True,
    }
    (out / f"{basename}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[success] image={image.shape} raw_masks={len(masks)} "
          f"regions={len(unique)} union={100 * raw_union.mean():.2f}%", flush=True)
    print(f"[saved] {out / (basename + '.npy')}", flush=True)
    print(f"[saved] {out / (basename + '.png')}", flush=True)
    print(f"[saved] {out / (basename + '.json')}", flush=True)
    print(f"[saved] {masks_path}", flush=True)


if __name__ == "__main__":
    main()
