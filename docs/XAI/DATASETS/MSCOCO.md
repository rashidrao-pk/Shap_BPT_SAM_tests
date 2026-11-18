# MS-COCO explanations

Run from the repository root on a GPU node. First [download the dataset](../../DATASETS/MS_COCO.md), get the checkpoint ([YOLO11s](../../MODELS/YOLO.md), [DETR](../../MODELS/DETR.md)), and [generate partitions](../../Partitions/DATASETS/MSCOCO.md). Make sure `data.mask_dir_final` names the generated folder.

## YOLO

```bash
python src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito --model yolo11s --show_gt --limit 1 \
  --max-evals 1000 --bg-type full --no-heatmaps
```

### SAM version and compact partitions

The same command for each SAM model (`--sam`, its folder from the generator) and each saved compact file (`--compact-strategy`). Without `--compact-strategy`, the config's default `split` is used. Results of `split` go to `<partition-folder>_split`; each run prints `Partitions: ... | compact file: ...` first.

```bash
# SAM-1 ViT-B, greedy compact partitions
python src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito --sam sam1 --compact-strategy greedy --model yolo11s --show_gt --limit 1 \
  --max-evals 1000 --bg-type full --no-heatmaps

# SAM-1 ViT-B, split compact partitions
python src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito --sam sam1 --compact-strategy split --model yolo11s --show_gt --limit 1 --show-gt \
  --max-evals 1000 --bg-type full --no-heatmaps

# SAM-2 Large, greedy compact partitions
python src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito --sam sam2 --compact-strategy greedy --model yolo11s --show_gt --limit 1 --show-gt \
  --max-evals 1000 --bg-type full --no-heatmaps

# SAM-2 Large, split compact partitions
python src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito --sam sam2 --compact-strategy split --model yolo11s --show_gt --limit 1 \
  --max-evals 1000 --bg-type full --explain_all --no-heatmaps
```

For all options, the output layout, and how the explained class is chosen, see the [YOLO guide](OD_yolo.md).

## DETR

```bash
python src/scripts/explanations/OD_DETR.py \
  --config MSCOCO_epito --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --local-files-only
```

Drop `--local-files-only` if the weights are not cached yet. DETR needs `transformers` and `timm`.

`--class-aggregation sum` is the default, so the score adds all detections of the class. `--class-aggregation max` uses the strongest one. `--threshold` defaults to 0, which keeps masked scores continuous; displayed boxes use at least 0.3.

## Ground truth

`--show-gt` shows COCO segmentation for the explained class. Classes are matched by name, because YOLO and COCO number them differently. If the class has no segmentation in the image, an empty mask is saved with an explicit status.

## Reading the results

- The score sums or maximizes over detections, so a map explains all detections of a class, not one object.
- val2017 is outside both detectors' COCO training set, but it is the standard validation split, so treat it as a benchmark rather than unseen data.
- PACO images overlap COCO. Do not explain them with these detectors as if they were unseen: see [PACO](../../DATASETS/PACO.md).

## Outputs and resume

```text
<output.explanations>/OD_DETR/<partition-folder>/bg_full_B1000/<image-id>/
<output.evaluations>/OD_DETR/<partition-folder>/bg_full_B1000/
```

Repeat the same command to resume: completed images are skipped. `--no-heatmaps` skips individual method images, and keeps the combined plots and numeric results.
