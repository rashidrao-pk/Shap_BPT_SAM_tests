# PACO: Parts and Attributes of Common Objects

[PACO](https://github.com/facebookresearch/paco) ([paper](https://arxiv.org/abs/2301.01795)) adds part masks for 75 object classes and 456 object-part classes on real photographs. It tests whether an explanation follows object parts in cluttered scenes, where PartImageNet's single centred objects are easier.

| | |
| --- | --- |
| Images | PACO-LVIS: 45,790 train, 2,410 val, 9,443 test (COCO images) |
| Ground truth | COCO-style polygons for whole objects and for parts, e.g. `mug:handle` |
| Model | [ResNet-50](../MODELS/RESNET50.md) |
| Download | 135 MB annotations; only the images of the chosen split, not the COCO zips |

## 1. Download the dataset

```bash
python src/scripts/download_datasets.py --dataset paco --assets dataset
```

This verifies the annotation archive against the SHA-256 published by the authors, then fetches each listed image from its `coco_url` into `/beegfs/home/mrashid/datasets/PACO/images/`. The val split is 2,410 images, so the 18 GB `train2017.zip` is not needed. Rerunning retries only missing images.

For another split, use `--split test` and change both `annotation_file` entries in `src/configs/PACO_epito.yaml`.

The PACO-EGO4D subset is not included: its frames need an [Ego4D licence](https://ego4ddataset.com) and are not part of this workflow.

## 2. Models

- [ResNet-50, shared with PartImageNet](../MODELS/RESNET50.md)

## 3. Next steps

1. [Generate partitions](../Partitions/DATASETS/PACO.md)
2. [Run explanations](../XAI/DATASETS/PACO.md)

## Limits on what you can claim

- Val images average 4.6 annotated objects, and only 789 of 2,410 have one. The default contour covers parts of every annotated object, not just the one the classifier predicted. For localization, prefer single-object images, or a `--gt-class` matched to the prediction.
- Not every object in a scene is annotated, so attribution outside the contour is not automatically wrong.
- ImageNet training images are not audited against COCO, so these are not guaranteed unseen.
