# Evaluate partitions before XAI

This check shows whether a dataset's partitions can express its ground truth, so you can decide whether the dataset is worth an XAI run. It needs no model, checkpoint, or GPU.

For each GT object or part, it finds the union of partition cells with the highest IoU, using the method in `notebooks/stats.ipynb`. It then does the same with a plain grid of the same size. A partition that barely beats a grid adds little structure.

## Run it

From the repository root, in your EPITO GPU session:

```bash
python src/scripts/evaluate_partitions.py --all-sam
```

This scores every completed partition with GT, for every dataset and every SAM folder. Each row names the model that made the partitions, read from the folder name: `SAM1-VB` (SAM-1 ViT-B; `-VL`, `-VH` for ViT-L, ViT-H), `SAM2-T` (SAM-2 tiny), `SAM2-S` (small), `SAM2-B+` (base plus) and `SAM2-L` (large). Plot file names use the same label, for example `BUSI_SAM2-L.png`. Common changes:

```bash
python src/scripts/evaluate_partitions.py                                    # Only the folder each config selects
python src/scripts/evaluate_partitions.py --config CAMO_epito PACO_epito     # Selected datasets
python src/scripts/evaluate_partitions.py --limit 100                        # Quick check on fewer images
python src/scripts/evaluate_partitions.py --no-plots                         # Tables only
```

## Outputs

Saved in `results/evaluations/partition_quality/`:

| File                                        | Contents                                                                                  | Thesis equivalent    |
| ------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------- |
| `conclusion.md`                             | Verdict and reason for each dataset                                                       |                      |
| `partition_quality_summary.csv`             | Mean, SD, and median of best IoU and GT coverage; share at each IoU level                 | Tables 4.8 and 4.10  |
| `partition_quality_objects.csv`             | Best IoU, grid IoU, and coverage for each GT object                                       |                      |
| `diagnosis.md`                              | For each dataset: why objects score low, by cause, size, and class, plus its worst images |                      |
| `partition_quality_images.csv`              | Every image, worst first, with its main cause                                             |                      |
| `failure_causes`                            | Share of objects below IoU 0.5, split by cause                                            |                      |
| `iou_by_object_size`                        | Median best IoU by object size                                                            |                      |
| `worst/<dataset>.png`                       | The worst images, set by `--worst` (default 6)                                            |                      |
| `overview`                                  | Median best IoU against the grid, the limits, and the verdict                             |                      |
| `best_iou_boxplot`, `best_iou_violin`       | Best IoU distribution; the triangle is the mean                                           | Figure 4.8           |
| `gt_coverage_boxplot`, `gt_coverage_violin` | GT coverage distribution                                                                  | Figure 4.9           |
| `iou_thresholds`                            | Share of objects at IoU 0.5, 0.75, and 0.9                                                | Table 4.10           |
| `examples/<dataset>.png`                    | Highest, middle, and lowest IoU images: GT, partition, optimal subsets, and masks → GT    | Figures 4.10 to 4.12 |

Plots are saved as PNG and SVG, except the examples, which are PNG. Datasets with too few images for a verdict are left out of the plots but stay in the tables.

## Verdicts

| Verdict           | Meaning                                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `USE`             | Median IoU at least 0.75, at least 75% of objects at IoU 0.5, and a median lead over the grid of at least 0.05 |
| `CAUTION`         | Usable, but at least one `USE` condition failed; the reason names it                                           |
| `SKIP`            | Median IoU below 0.5, or no lead over the grid                                                                 |
| `NEED PARTITIONS` | Fewer than 30 partitions, so no verdict is given                                                               |
| `NO GT`           | Fewer than 30 evaluated images have GT                                                                         |
| `ERROR`           | The dataset could not be read; the reason gives the error                                                      |

The limits are starting values, set just below the MS-COCO reference from the notebook: median IoU 0.825 and 79.3% of objects at IoU 0.5. They have not been checked against XAI results. Change them with `--use-median-iou`, `--use-iou-50`, `--skip-median-iou`, `--min-gain`, and `--min-images`.

## Why objects score low

An object below IoU 0.5 gets one cause:

| Cause        | Meaning                                                           | What to check                                                           |
| ------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `Unassigned` | At least half of the object is in cell 0, which is never selected | SAM found no mask there, as with dark, low-contrast regions             |
| `Merged`     | The selected cells are mostly outside the object                  | The object has no boundary of its own, which is common for tiny objects |
| `Partial`    | The cells that fit cover only part of the object                  | The object is split among cells that are mostly outside it              |

Sizes are the object's share of the image: tiny below 0.1%, small below 1%, medium below 10%, and large above that.

CelebA parts are combined into one face object by default. Add `--split-parts` to score each part alone. A face covers most of the image, so even a plain grid scores high there.

If an image cannot be read, it is skipped and counted, and the verdict reason says how many.

## Reading the results

- `USE` means that good regions exist. It does not show that the model relies on them, or that attributions will match the GT.
- Part datasets contain many small GT objects, so they score lower than whole-object datasets. A low score there can reflect the task rather than poor partitions.
- High coverage with low IoU means one large cell contains the object, so the partition does not follow its boundary. Check `examples/` for this.
- `Coverage` is the share of the GT inside the best union of cells. Cell 0 is background and is never selected.
- Results for COCO use each non-crowd annotation, as in the notebook. Other datasets use each label in the GT map, so overlapping parts count only where they are visible.
