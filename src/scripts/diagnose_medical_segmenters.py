#!/usr/bin/env python3
"""Diagnose MedSAM / MedicoSAM against SAM-1 ViT-B on one target: a synthetic bright disk, or a real image
with its mask (e.g. a BUSI image and its *_mask.png).

For each model it reports, against the target:
  1. direct prompts with no filtering: one box around the target, and one point at its centre
     (MedSAM: box only, as trained): IoU, predicted IoU, stability;
  2. the automatic generator used for partitions, with filters off (thresholds 0) and at the defaults (0.6):
     mask count, best IoU, and the spread of predicted IoU and stability over all candidates.
If the direct prompt is good but the automatic run loses it, the filters are the problem; if the direct prompt
is already bad, it is the model, the preprocessing or the test image. SAM-1 ViT-B is the control.

    python src/scripts/diagnose_medical_segmenters.py --device cuda
    python src/scripts/diagnose_medical_segmenters.py --device cuda \
      --image "/beegfs/home/mrashid/datasets/BUSI/benign/benign (1).png" \
      --mask "/beegfs/home/mrashid/datasets/BUSI/benign/benign (1)_mask.png"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import medical_segmenters as med  # noqa: E402

CHECKPOINTS = {**med.DEFAULT_CHECKPOINTS, 'sam1': ROOT / 'checkpoints/sam1/sam_vit_b_01ec64.pth'}


def disk_target(size=256, radius=40):
    yy, xx = np.mgrid[:size, :size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius ** 2
    image = np.full((size, size, 3), 30, dtype=np.uint8)
    image[disk] = 220
    return image, disk


def iou(mask, target):
    union = np.logical_or(mask, target).sum()
    return float(np.logical_and(mask, target).sum() / union) if union else 0.


def target_box(target, margin=.1):
    ys, xs = np.nonzero(target)
    height, width = target.shape
    pad_y, pad_x = margin * (ys.max() - ys.min() + 1), margin * (xs.max() - xs.min() + 1)
    return np.array([max(0, xs.min() - pad_x), max(0, ys.min() - pad_y),
                     min(width, xs.max() + 1 + pad_x), min(height, ys.max() + 1 + pad_y)])


def spread(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return 'n/a'
    q = np.percentile(values, [0, 25, 50, 75, 100])
    return 'min {:.2f} | q25 {:.2f} | median {:.2f} | q75 {:.2f} | max {:.2f}'.format(*q)


def medsam_direct(model, device, image, box):
    import torch
    import torch.nn.functional as F
    height, width = image.shape[:2]
    tensor = torch.from_numpy(med.medsam_preprocess(image)).permute(2, 0, 1)[None].to(device)
    box_1024 = torch.as_tensor(box / np.array([width, height, width, height]) * 1024, dtype=torch.float32,
                               device=device)[None, None]
    with torch.no_grad():
        embedding = model.image_encoder(tensor)
        sparse, dense = model.prompt_encoder(points=None, boxes=box_1024, masks=None)
        logits, predicted = model.mask_decoder(image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
                                               sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                               multimask_output=False)
    stability = float(med.stability_scores(logits[:, 0])[0])
    mask = (F.interpolate(torch.sigmoid(logits), size=(height, width), mode='bilinear',
                          align_corners=False)[0, 0] > .5).cpu().numpy()
    return mask, float(predicted[0, 0]), stability


def sam_direct(model, image, box, point):
    from segment_anything import SamPredictor
    predictor = SamPredictor(model)
    predictor.set_image(image)
    results = {}
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    results['box'] = (masks[0], float(scores[0]))
    masks, scores, _ = predictor.predict(point_coords=point[None], point_labels=np.array([1]), multimask_output=True)
    best = int(np.argmax(scores))
    results['point (best of 3)'] = (masks[best], float(scores[best]))
    return results


def automatic(name, model, device, image, threshold):
    if name == 'medsam':
        generator = med.MedSAMBoxGridGenerator(model, device, pred_iou_thresh=threshold,
                                               stability_score_thresh=threshold)
    else:
        from segment_anything import SamAutomaticMaskGenerator
        generator = SamAutomaticMaskGenerator(model, points_per_side=32, pred_iou_thresh=threshold,
                                              stability_score_thresh=threshold, box_nms_thresh=.7)
    return generator.generate(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=['sam1', 'medsam', 'medicosam'], choices=list(CHECKPOINTS))
    parser.add_argument('--image', type=Path, help='Real image (default: synthetic bright disk)')
    parser.add_argument('--mask', type=Path, help='Target mask for --image (non-zero = target)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if bool(args.image) != bool(args.mask):
        parser.error('--image and --mask go together')
    if args.image:
        image = np.asarray(Image.open(args.image).convert('RGB'))
        target = np.asarray(Image.open(args.mask).convert('L')) > 0
        if target.shape != image.shape[:2] or not target.any():
            parser.error('mask must match the image size and contain the target')
        source = args.image.name
    else:
        image, target = disk_target()
        source = 'synthetic disk (256 x 256, radius 40)'
    box = target_box(target)
    ys, xs = np.nonzero(target)
    point = np.array([xs.mean(), ys.mean()])
    print(f'Target: {source}; {int(target.sum())} px ({100 * target.mean():.1f}% of the image); '
          f'box {box.round(1).tolist()}\n')
    for name in args.models:
        checkpoint = CHECKPOINTS[name]
        print(f'=== {name} ({checkpoint}) ===')
        if not checkpoint.is_file():
            print('  checkpoint missing, skipped\n')
            continue
        model, _ = med.load_sam_vit_b(checkpoint, args.device)
        if name == 'medsam':
            mask, predicted, stability = medsam_direct(model, args.device, image, box)
            print(f'  direct box prompt: IoU {iou(mask, target):.3f} | predicted IoU {predicted:.3f} | '
                  f'stability {stability:.3f} | mask {100 * mask.mean():.1f}% of the image')
        else:
            for prompt, (mask, score) in sam_direct(model, image, box, point).items():
                print(f'  direct {prompt}: IoU {iou(mask, target):.3f} | predicted IoU {score:.3f} | '
                      f'mask {100 * mask.mean():.1f}% of the image')
        for threshold in (0., .6):
            annotations = automatic(name, model, args.device, image, threshold)
            best = max((iou(item['segmentation'], target) for item in annotations), default=0.)
            print(f'  automatic, thresholds {threshold:.1f}: {len(annotations)} masks | best IoU {best:.3f}')
            if threshold == 0.:
                print(f"    predicted IoU: {spread([a['predicted_iou'] for a in annotations])}")
                print(f"    stability:     {spread([a['stability_score'] for a in annotations])}")
        print()


if __name__ == '__main__':
    main()
