# PACO: Parts and Attributes of Common Objects

[PACO](https://github.com/facebookresearch/paco) ([paper](https://arxiv.org/abs/2301.01795)) adds part masks for 75 object classes and 456 object-part classes on real photographs. It tests whether an explanation follows object parts in cluttered scenes, where PartImageNet's single centred objects are easier.

| | |
| --- | --- |
| Images | PACO-LVIS: 45,790 train, 2,410 val, 9,443 test (COCO images) |
| Ground truth | COCO-style polygons for whole objects and for parts, e.g. `mug:handle` |
| Classifier | Frozen ImageNet ResNet-50, shared with PartImageNet |
| Download | 135 MB annotations; only the images of the chosen split, not the COCO zips |

## 1. Download the dataset and checkpoint

```bash
python src/scripts/download_datasets.py --dataset paco
```

This verifies the annotation archive against the SHA-256 published by the authors, then fetches each listed image from its `coco_url` into `/beegfs/home/mrashid/datasets/PACO/images/`. The val split is 2,410 images, so the 18 GB `train2017.zip` is not needed. Rerunning retries only missing images.

For another split, use `--split test` and change both `annotation_file` entries in `src/configs/PACO_epito.yaml`.

The PACO-EGO4D subset is not included: its frames need an [Ego4D licence](https://ego4ddataset.com) and are not part of this workflow.

## 2. Why this checkpoint

The download reuses `checkpoints/partimagenet/resnet50-11ad3fa6.pth` (torchvision `IMAGENET1K_V2`).

- **Not a COCO detector.** PACO-LVIS images are COCO images, and all 2,410 val images come from `train2017`, which YOLO and DETR were trained on. An ImageNet classifier did not see them through COCO.
- **Not PACO's own models.** The [model zoo](https://github.com/facebookresearch/paco/blob/main/docs/MODEL_ZOO.md) holds detection models that need detectron2 and predict boxes, not one class score. They do not fit this runner.

The cost is that ImageNet's 1000 classes are not PACO's 75, so the explained class often differs from the annotated object.

## 3. Generate partitions

```bash
python src/scripts/generate_paco_partitions.py \
  --config PACO_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda --max-regions 62 --generate_at_least_n_masks 50 \
  --save-partitions sam compact
```

## 4. Explain

```bash
python src/scripts/explanations/OC_PACO.py \
  --config PACO_epito --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --explain_all
```

`--show-gt` outlines all annotated **parts**. Use `--gt-class "mug:handle"` for one part or `--gt-class mug` for whole objects. Names are in the `categories` list of the annotation JSON.

## Limits on what you can claim

- Val images average 4.6 annotated objects, and only 789 of 2,410 have one. The default contour covers parts of every annotated object, not just the one the classifier predicted. For localization, prefer single-object images, or a `--gt-class` matched to the prediction.
- Not every object in a scene is annotated, so attribution outside the contour is not automatically wrong.
- ImageNet training images are not audited against COCO, so these are not guaranteed unseen.
