#!/usr/bin/env python3
"""Generate SAM1/SAM2 automatic masks and disjoint ShapBPT-ready region maps.

Dependencies: numpy, scipy, pillow, matplotlib, torch, and SAM1/SAM2 as needed.
Runs from any working directory; use --raw-masks to refine saved SAM masks offline.
"""
from __future__ import annotations

import argparse
import gc
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
    p.add_argument("--strategy", choices=["refined", "raw_overwrite", "compact"], default="refined")
    p.add_argument("--output-dir", type=Path, default=Path("results/partitions/sam2"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--sam2-config", default=None,
                   help="Hydra config name, e.g. configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--sam1-variant", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    # SAM automatic mask generator parameters; see SAM docs for details.
    p.add_argument("--points-per-side", type=int, default=32)
    p.add_argument("--points-per-batch", type=int, default=64)
    p.add_argument("--pred-iou-thresh", type=float, default=0.6)
    # 
    p.add_argument("--stability-score-thresh", type=float, default=0.6)
    p.add_argument("--stability-score-offset", type=float, default=1.0)
    p.add_argument("--mask-threshold", type=float, default=0.0)
    
    p.add_argument("--box-nms-thresh", type=float, default=0.7)
    p.add_argument("--crop-n-layers", type=int, default=0)
    p.add_argument("--crop-nms-thresh", type=float, default=0.7)
    p.add_argument("--crop-overlap-ratio", type=float, default=512/1500)
    p.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    p.add_argument("--point-grids", type=float, default=None)
    p.add_argument("--min-mask-region-area", type=float, default=0)
    p.add_argument("--output-mode", type=str, default="binary_mask")
    p.add_argument("--use-m2m", type=bool, default=True)

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


def build_generator(args):
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
            # new parameters for SAM
            stability_score_offset=args.stability_score_offset,
            mask_threshold=args.mask_threshold,
            box_nms_thresh=args.box_nms_thresh,
            crop_n_layers=args.crop_n_layers,
            crop_nms_thresh=args.crop_nms_thresh,
            crop_overlap_ratio=args.crop_overlap_ratio,
            crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
            point_grids=args.point_grids,
            min_mask_region_area=args.min_mask_region_area,
            output_mode=args.output_mode,
            use_m2m=args.use_m2m,

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
            stability_score_offset=args.stability_score_offset,
            box_nms_thresh=args.box_nms_thresh,
            crop_n_layers=args.crop_n_layers,
            crop_nms_thresh=args.crop_nms_thresh,
            crop_overlap_ratio=args.crop_overlap_ratio,
            crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
            point_grids=args.point_grids,
            min_mask_region_area=args.min_mask_region_area,
            output_mode=args.output_mode,
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
    return generator, device, str(checkpoint)


def generate_at_least_n_masks(image, generator, min_masks=400, max_steps=3,
                             step=0.2, min_pred_iou_thresh=0.1,
                             min_stability_score_thresh=0.1):
    """Notebook threshold search; reuse the model and reset settings per image."""
    if min_masks < 1 or max_steps < 1 or step <= 0:
        raise ValueError('Mask target, attempt count, and tuning step must be positive')
    original = generator.pred_iou_thresh, generator.stability_score_thresh
    history = []
    try:
        for attempt in range(1, max_steps + 1):
            #generation and timing
            start = time.perf_counter()
            annotations = generator.generate(image)
            elapsed = time.perf_counter() - start
            history.append({
                'step': attempt, 'pred_iou_thresh': generator.pred_iou_thresh,
                'stability_score_thresh': generator.stability_score_thresh,
                'n_masks': len(annotations), 'time_sam': elapsed,
            })

            print(f'[mask tuning] step={attempt}/{max_steps} '
                  f'iou={generator.pred_iou_thresh:.2f} '
                  f'stability={generator.stability_score_thresh:.2f} '
                  f'masks={len(annotations)}/{min_masks} seconds={elapsed:.3f}', flush=True)

            # Check if we have enough masks
            if len(annotations) >= min_masks:
                break
            # Lower thresholds to generate more masks, but never below the minimums.
            iou = max(min_pred_iou_thresh, generator.pred_iou_thresh - step)
            stability = max(min_stability_score_thresh, generator.stability_score_thresh - step)
            
            # Never increase an already lower user-specified threshold.
            iou = min(generator.pred_iou_thresh, iou)
            stability = min(generator.stability_score_thresh, stability)
            if (iou, stability) == (generator.pred_iou_thresh, generator.stability_score_thresh):
                break
            generator.pred_iou_thresh, generator.stability_score_thresh = iou, stability
        print(f'[mask tuning] target_met={len(annotations) >= min_masks} '
              f'attempts={len(history)} masks={len(annotations)}', flush=True)
        return annotations, history
    finally:
        generator.pred_iou_thresh, generator.stability_score_thresh = original


def infer_masks(image, args, loaded_generator=None, return_history=False, return_annotations=False):
    import torch
    generator, device, checkpoint = loaded_generator or build_generator(args)
    t0 = time.perf_counter()
    history = []
    with torch.inference_mode():
        while True:
            try:
                target = getattr(args, 'generate_at_least_n_masks', None)
                if target:
                    annotations, history = generate_at_least_n_masks(
                        image, generator, min_masks=target,
                        max_steps=args.mask_tuning_max_steps,
                    )
                else:
                    annotations = generator.generate(image)
                break
            except torch.OutOfMemoryError:
                batch_size = getattr(generator, 'points_per_batch', 1)
                if str(device).split(':')[0] != 'cuda' or batch_size <= 1:
                    raise
                smaller_batch = max(1, batch_size // 2)
                print(f'[CUDA OOM] image={image.shape[:2]} '
                      f'points_per_batch={batch_size} -> {smaller_batch}; '
                      'retrying this image', flush=True)
            # Outside the exception handler: release the failed call's traceback
            # and cached image features before reclaiming unused CUDA blocks.
            predictor = getattr(generator, 'predictor', None)
            if predictor is not None and hasattr(predictor, 'reset_predictor'):
                predictor.reset_predictor()
            gc.collect()
            torch.cuda.empty_cache()
            generator.points_per_batch = smaller_batch
            # Keep the smaller batch for subsequent images. CLI settings stay
            # unchanged so completed outputs remain resumable.
    dt = time.perf_counter() - t0
    masks = np.stack([m["segmentation"] for m in annotations]).astype(bool) if annotations else np.empty((0, *image.shape[:2]), dtype=bool)
    result = masks, dt, device, str(checkpoint)
    if return_annotations:
        return (*result, history, annotations)
    return (*result, history) if return_history else result


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

def select_masks_by_unpicked_area(masks, predicted_ious, n_masks_to_select):
    """
    Iteratively selects masks using the following score:

        score = unpicked_area * predicted_iou

    At each iteration, recalculates the scores of the remaining masks
    and selects the one with the highest score.

    Stops the selection process when:
    - n_masks_to_select masks have been selected;
    - no masks remain;
    - no remaining mask adds any new pixels.

    Returns the selected masks in the order they were selected.
    """

    masks = np.asarray(masks, dtype=bool)
    predicted_ious = np.asarray(predicted_ious, dtype=np.float32)

    if masks.ndim != 3:
        raise ValueError("masks deve avere forma (N, H, W)")

    n, h, w = masks.shape

    if predicted_ious.shape != (n,):
        raise ValueError("Serve un predicted_iou per ogni maschera")

    if n_masks_to_select < 1:
        raise ValueError("n_masks_to_select deve essere >= 1")

    if n == 0:
        return masks.copy()

    picked_pixels = np.zeros((h, w), dtype=bool)

    remaining_indices = list(range(n))
    selected_indices = []

    max_selections = min(n_masks_to_select, n)

    for _ in range(max_selections):

        best_index = None
        best_score = -1.0
        best_unpicked_area = 0

        # Ricalcola lo score di tutte le maschere disponibili
        for idx in remaining_indices:

            unpicked_mask = masks[idx] & (~picked_pixels)

            unpicked_area = int(
                np.count_nonzero(unpicked_mask)
            )

            score = (
                unpicked_area * float(predicted_ious[idx])
            )

            if score > best_score:
                best_index = idx
                best_score = score
                best_unpicked_area = unpicked_area

        # Nessuna maschera aggiunge nuovi pixel
        if best_index is None or best_unpicked_area == 0:
            break

        # Salva la maschera migliore
        selected_indices.append(best_index)

        # Aggiorna la copertura
        picked_pixels |= masks[best_index]

        # Rimuove la maschera selezionata dalle candidate
        remaining_indices.remove(best_index)

    return masks[selected_indices]

def compact_partitions(masks, predicted_ious, max_regions):
    selected_masks = select_masks_by_unpicked_area(
        masks,
        predicted_ious,
        n_masks_to_select=max_regions
    )

    partitions = raw_overwrite(selected_masks)

    return partitions.astype(np.uint16)



def main():
    args = parse_args()
    if args.max_regions < 1:
        raise ValueError("--max-regions must be >=1")
    image = np.asarray(Image.open(args.image).convert("RGB"))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    predicted_ious = None
    if args.raw_masks:
        with np.load(args.raw_masks, allow_pickle=False) as loaded:
            masks = loaded["masks"].astype(bool)

            if "predicted_ious" in loaded:
                predicted_ious = loaded["predicted_ious"].astype(np.float32)

        inference_seconds, device, checkpoint = None, "cached", None
    elif args.strategy == "compact":
            (
                masks,
                inference_seconds,
                device,
                checkpoint,
                history,
                annotations
            ) = infer_masks(
                image,
                args,
                return_annotations=True
            )

            predicted_ious = np.asarray(
                [ann["predicted_iou"] for ann in annotations],
                dtype=np.float32
            )
    else:
        masks, inference_seconds, device, checkpoint = infer_masks(
            image,
            args
        )
    if args.strategy == "compact" and predicted_ious is None:
        raise ValueError(
            "The compact strategy requires predicted_ious, "
            "but they are missing from the loaded raw masks file."
        )

    if masks.ndim != 3 or masks.shape[1:] != image.shape[:2]:
        raise ValueError(f"Expected masks [N,{image.shape[0]},{image.shape[1]}], got {masks.shape}")
    masks_path = out / f"{stem}_{args.model}_raw_masks.npz"

    if predicted_ious is not None:
        np.savez_compressed(
            masks_path,
            masks=masks,
            predicted_ious=predicted_ious
        )
    else:
        np.savez_compressed(
            masks_path,
            masks=masks
        )

    raw_union = np.any(masks, axis=0) if len(masks) else np.zeros(image.shape[:2], dtype=bool)
    t0 = time.perf_counter()
    if args.strategy == "refined":
        partitions = refined_partitions(masks, args.min_region_pixels, args.max_regions)
    elif args.strategy == "compact":
        
        partitions = compact_partitions(masks, predicted_ious, args.max_regions)
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
