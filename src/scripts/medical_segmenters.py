"""Medical segmentation priors for the partition generators: MedSAM and MedicoSAM (TO_DO T2).

Both are SAM ViT-B models fine-tuned on medical images. They return the same annotation dicts as SAM's
automatic mask generator (`segmentation`, `area`, `bbox`, `predicted_iou`, `stability_score`, ...), so the
existing mask-count tuning, compact selection, region cap and saved metadata work unchanged.

MedSAM (Ma et al., Nat. Commun. 2024; github.com/bowang-lab/MedSAM, Apache-2.0)
    Fine-tuned with box prompts only, so SAM's point-grid "everything" mode is out of distribution for it.
    `MedSAMBoxGridGenerator` prompts it with a multi-scale grid of square boxes instead and follows the
    official `MedSAM_Inference.py`: resize to 1024 x 1024 (bicubic, anti-aliased), min-max normalise to
    [0, 1], `image_encoder` once per image, then `prompt_encoder(boxes=...)` and
    `mask_decoder(multimask_output=False)`; sigmoid, bilinear upsampling to the image size, threshold 0.5.
    `--medsam-prompt point-grid` runs SAM's standard point-grid generator with MedSAM weights instead; that
    ablation is out of distribution and labelled so.

MedicoSAM (Archit, Freckmann, Pape 2025; github.com/computational-cell-analytics/medico-sam, MIT)
    Fine-tuned with point and box prompts in micro_sam, which keeps SAM's preprocessing. Its ViT-B weights
    are loaded into segment_anything's `SamAutomaticMaskGenerator`, so it runs exactly like SAM-1 ViT-B
    with different weights. This avoids depending on micro_sam; `segmenter_info['path']` records it.
"""
import hashlib
from pathlib import Path

import numpy as np

MEDSAM_INPUT_SIZE = 1024
BOX_SIDE_FRACTIONS = (1 / 2, 1 / 4, 1 / 8)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = {
    'medsam': ROOT / 'checkpoints/medsam/medsam_vit_b.pth',
    'medicosam': ROOT / 'checkpoints/medicosam/vit_b_medicosam.pt',
}


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def read_state_dict(path):
    """SAM ViT-B weights from a plain state dict or a training checkpoint (`model_state`, `model`,
    `state_dict`), with `module.` / `sam.` prefixes removed. Returns (state dict, how it was found)."""
    import torch
    info = {}
    try:
        loaded = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:  # Training checkpoints may hold other pickled objects; the file is from the official source.
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        info['full_pickle'] = True
    for key in ('model_state', 'model', 'state_dict'):
        if isinstance(loaded, dict) and isinstance(loaded.get(key), dict):
            loaded, info['nested_key'] = loaded[key], key
            break
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f'{path}: no state dict found')
    for prefix in ('module.', 'sam.'):
        if all(name.startswith(prefix) for name in loaded):
            loaded = {name[len(prefix):]: value for name, value in loaded.items()}
            info.setdefault('stripped_prefixes', []).append(prefix)
    return loaded, info


def load_sam_vit_b(checkpoint, device):
    """A segment_anything ViT-B model holding the given fine-tuned weights (strict key match)."""
    from segment_anything import sam_model_registry
    model = sam_model_registry['vit_b'](checkpoint=None)
    state, info = read_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), info


def box_grid(height, width, fractions=BOX_SIDE_FRACTIONS):
    """Square boxes (x0, y0, x1, y1) in pixels: side = fraction x shorter image side, stride = side / 2,
    plus a last row and column flush with the bottom and right edges. A square image gives 9 + 49 + 225."""
    shorter = min(height, width)
    boxes = []
    for fraction in fractions:
        side = max(2, int(round(shorter * fraction)))
        stride = max(1, side // 2)

        def starts(extent):
            values = list(range(0, extent - side + 1, stride))
            if values[-1] != extent - side:
                values.append(extent - side)
            return values

        boxes.extend([x, y, x + side, y + side] for y in starts(height) for x in starts(width))
    return np.unique(np.asarray(boxes, dtype=np.float64), axis=0)


def medsam_preprocess(image):
    """MedSAM_Inference.py preprocessing: grey to 3 channels, bicubic anti-aliased resize to 1024 x 1024
    (cast back to uint8), then min-max normalisation to [0, 1]. Returns float32 (1024, 1024, 3)."""
    from skimage import transform
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=-1)
    resized = transform.resize(image, (MEDSAM_INPUT_SIZE, MEDSAM_INPUT_SIZE), order=3,
                               preserve_range=True, anti_aliasing=True).astype(np.uint8)
    resized = (resized - resized.min()) / np.clip(resized.max() - resized.min(), a_min=1e-8, a_max=None)
    return resized.astype(np.float32)


def stability_scores(logits, threshold=0.0, offset=1.0):
    """SAM's stability score: IoU of the masks thresholded at threshold +/- offset, per mask."""
    inner = (logits > threshold + offset).flatten(1).sum(1).float()
    outer = (logits > threshold - offset).flatten(1).sum(1).float()
    return inner / outer.clamp(min=1)


class MedSAMBoxGridGenerator:
    """SAM-style automatic mask generator for MedSAM, prompted with a multi-scale box grid.

    The attribute names follow SAM's generator, so `generate_at_least_n_masks` can relax
    `pred_iou_thresh` / `stability_score_thresh` and the OOM fallback can halve `points_per_batch`
    (here: boxes per decoder batch). Decoder outputs are cached for the last image, so relaxing the
    thresholds only refilters and costs no extra model calls.

    Note: MedSAM was trained with mask losses only, so its IoU head is SAM's, applied to fine-tuned
    features. Predicted IoU is kept for parity with SAM (filter and compact-selection weight); the
    stability score is the better-founded criterion. Both are saved per mask.
    """

    def __init__(self, model, device, pred_iou_thresh=0.6, stability_score_thresh=0.6, stability_score_offset=1.0,
                 box_nms_thresh=0.7, points_per_batch=64, side_fractions=BOX_SIDE_FRACTIONS):
        self.model, self.device = model, device
        self.pred_iou_thresh, self.stability_score_thresh = pred_iou_thresh, stability_score_thresh
        self.stability_score_offset, self.box_nms_thresh = stability_score_offset, box_nms_thresh
        self.points_per_batch, self.side_fractions = points_per_batch, tuple(side_fractions)
        self._cache_key, self._cache = None, None

    def _candidates(self, image):
        import torch
        key = (image.shape, hashlib.blake2b(np.ascontiguousarray(image).tobytes(), digest_size=16).hexdigest())
        if key == self._cache_key:
            return self._cache
        height, width = image.shape[:2]
        tensor = torch.from_numpy(medsam_preprocess(image)).permute(2, 0, 1)[None].to(self.device)
        boxes = box_grid(height, width, self.side_fractions)
        scale = np.array([width, height, width, height], dtype=np.float64)
        boxes_1024 = boxes / scale * MEDSAM_INPUT_SIZE
        logits, ious = [], []
        with torch.no_grad():
            embedding = self.model.image_encoder(tensor)
            dense_pe = self.model.prompt_encoder.get_dense_pe()
            for start in range(0, len(boxes_1024), max(1, int(self.points_per_batch))):
                chunk = torch.as_tensor(boxes_1024[start:start + int(self.points_per_batch)],
                                        dtype=torch.float32, device=self.device)[:, None, :]
                sparse, dense = self.model.prompt_encoder(points=None, boxes=chunk, masks=None)
                low_res, iou = self.model.mask_decoder(
                    image_embeddings=embedding, image_pe=dense_pe, sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense, multimask_output=False)
                logits.append(low_res[:, 0])
                ious.append(iou[:, 0])
        logits = torch.cat(logits)
        self._cache_key = key
        self._cache = {'boxes': boxes, 'logits': logits, 'iou': torch.cat(ious).float().cpu().numpy(),
                       'shape': (height, width)}
        return self._cache

    def generate(self, image):
        import torch
        import torch.nn.functional as F
        from torchvision.ops import nms
        image = np.asarray(image)
        candidates = self._candidates(image)
        logits, iou = candidates['logits'], candidates['iou']
        height, width = candidates['shape']
        stability = stability_scores(logits, 0.0, self.stability_score_offset).cpu().numpy()
        low = (logits > 0).cpu().numpy()
        keep = np.flatnonzero((iou >= self.pred_iou_thresh) & (stability >= self.stability_score_thresh)
                              & low.any(axis=(1, 2)))
        if keep.size == 0:
            return []
        # Box NMS on the masks' own boxes (scaled from the 256 x 256 low-res masks), as SAM's generator does.
        size = low.shape[-1]
        rows, columns = low[keep].any(axis=2), low[keep].any(axis=1)
        mask_boxes = np.array([[np.argmax(c), np.argmax(r), size - np.argmax(c[::-1]), size - np.argmax(r[::-1])]
                               for r, c in zip(rows, columns)], dtype=np.float32)
        mask_boxes *= np.array([width, height, width, height], dtype=np.float32) / size
        survivors = nms(torch.from_numpy(mask_boxes), torch.from_numpy(iou[keep].astype(np.float32)),
                        self.box_nms_thresh).numpy()
        keep = keep[survivors]
        annotations = []
        for start in range(0, len(keep), 16):
            index = keep[start:start + 16]
            probabilities = F.interpolate(torch.sigmoid(logits[index])[:, None].float(), size=(height, width),
                                          mode='bilinear', align_corners=False)[:, 0]
            for number, mask in zip(index, (probabilities > 0.5).cpu().numpy()):
                if not mask.any():
                    continue
                ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
                x0, y0, x1, y1 = candidates['boxes'][number]
                annotations.append({
                    'segmentation': mask, 'area': int(mask.sum()),
                    'bbox': [int(xs[0]), int(ys[0]), int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1)],
                    'predicted_iou': float(iou[number]), 'stability_score': float(stability[number]),
                    'point_coords': [[float((x0 + x1) / 2), float((y0 + y1) / 2)]],
                    'prompt_box': [float(x0), float(y0), float(x1), float(y1)],
                    'crop_box': [0, 0, int(width), int(height)],
                })
        return annotations


def build_medical_generator(args, device, amg_kwargs):
    """(generator, checkpoint path) for --model medsam | medicosam. `amg_kwargs` are the SAM generator settings
    used for point-grid runs. The generator carries `segmenter_info`, which is saved in each image's metadata."""
    checkpoint = Path(args.checkpoint) if getattr(args, 'checkpoint', None) else DEFAULT_CHECKPOINTS[args.model]
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{args.model} checkpoint missing: {checkpoint}. '
                                'See docs/Partitions/SAM/MEDICAL_SEGMENTERS.md, or pass --checkpoint PATH')
    model, load_info = load_sam_vit_b(checkpoint, device)
    prompt = getattr(args, 'medsam_prompt', 'box-grid') if args.model == 'medsam' else 'point-grid'
    info = {'segmenter': args.model, 'backbone': 'vit_b', 'checkpoint_sha256': sha256(checkpoint),
            'prompt_mode': prompt, 'state_dict': load_info}
    if args.model == 'medsam' and prompt == 'box-grid':
        generator = MedSAMBoxGridGenerator(
            model, device, pred_iou_thresh=args.pred_iou_thresh, stability_score_thresh=args.stability_score_thresh,
            stability_score_offset=args.stability_score_offset, box_nms_thresh=args.box_nms_thresh,
            points_per_batch=args.points_per_batch)
        info.update(path='MedSAMBoxGridGenerator (MedSAM_Inference.py preprocessing)', in_distribution=True,
                    box_side_fractions=list(BOX_SIDE_FRACTIONS), box_stride='side/2')
    else:
        from segment_anything import SamAutomaticMaskGenerator
        generator = SamAutomaticMaskGenerator(model, **amg_kwargs)
        info.update(path='segment_anything.SamAutomaticMaskGenerator (SAM preprocessing)',
                    in_distribution=args.model == 'medicosam')
    generator.segmenter_info = info
    return generator, checkpoint
