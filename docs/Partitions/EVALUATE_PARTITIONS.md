# Evaluate partitions before XAI

This check shows whether a dataset's partitions can express its ground truth, so you can decide whether the dataset is worth an XAI run. It needs no model, checkpoint, or GPU.

For each GT object or part, it finds the union of partition cells with the highest IoU, using the method in `notebooks/stats.ipynb`. It then does the same with a plain grid of the same size. A partition that barely beats a grid adds little structure.

## Run it

From the repository root, in your EPITO environment:

```bash
python src/scripts/evaluate_partitions.py --all-sam
```

This scores up to 300 images with GT from each dataset, for every SAM partition folder. Common changes:

```bash
python src/scripts/evaluate_partitions.py                                    # Only the folder each config selects
python src/scripts/evaluate_partitions.py --config CAMO_epito PACO_epito     # Selected datasets
python src/scripts/evaluate_partitions.py --limit 1000                       # More images
```

It prints a table and saves these in `results/evaluations/partition_quality/`:

| File | Contents |
| --- | --- |
| `conclusion.md` | The verdict for each dataset, with the reason |
| `partition_quality_summary.csv` | All summary numbers |
| `partition_quality_objects.csv` | Best IoU, grid IoU, and coverage for each GT object |

## Verdicts

| Verdict | Meaning |
| --- | --- |
| `USE` | Median IoU at least 0.75, at least 75% of objects at IoU 0.5, and a median lead over the grid of at least 0.05 |
| `CAUTION` | Usable, but at least one `USE` condition failed; the reason names it |
| `SKIP` | Median IoU below 0.5, or no lead over the grid |
| `NEED PARTITIONS` | Fewer than 30 partitions, so no verdict is given |
| `NO GT` | Fewer than 30 evaluated images have GT |
| `ERROR` | The dataset could not be read; the reason gives the error |

The limits are starting values, set just below the MS-COCO reference from the notebook: median IoU 0.825 and 79.3% of objects at IoU 0.5. They have not been checked against XAI results. Change them with `--use-median-iou`, `--use-iou-50`, `--skip-median-iou`, `--min-gain`, and `--min-images`.

## Reading the results

- `USE` means that good regions exist. It does not show that the model relies on them, or that attributions will match the GT.
- Part datasets contain many small GT objects, so they score lower than whole-object datasets. A low score there can reflect the task rather than poor partitions.
- `Coverage` is the share of the GT inside the best union of cells. Cell 0 is background and is never selected.
- Results for COCO use each non-crowd annotation, as in the notebook. Other datasets use each label in the GT map, so overlapping parts count only where they are visible.
