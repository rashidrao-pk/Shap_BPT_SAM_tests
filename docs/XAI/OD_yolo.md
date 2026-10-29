# YOLO explanations from MS-COCO partitions

[OD_yolo.py](../../src/scripts/explations/OD_yolo.py) is a separate copy of
`run_yolo_full_all.py`. The original is unchanged. It retains YOLO prediction,
ShapBPT/region Kernel SHAP, deletion/insertion AUC, faithfulness metrics,
per-image heatmaps, plots, CSV summaries, resume checks, and the HTML report.

Use the same EPITO environment as [EPITO.md](EPITO.md#1-start-a-session).
No SAM inference or partition generation is run by this script.

Result keys, plot labels, and method-specific output paths use `SAM_c` and
`SAM_c_kernelshap`. Existing partition storage remains `*_compact.npy`; no
regeneration or renaming is required. The generator still uses
`--save-partitions compact` to produce this input.

## 1. Select the saved partition dataset

In `src/configs/MSCOCO_epito.yaml`, check image/annotation paths and set:

```yaml
data:
  masks_path: /beegfs/home/mrashid/datasets/
  mask_dir: SAM_BPT
  mask_dir_final: MSCOCO_sam2_large_62_refined_200
```

These are selected fields, not a replacement config. Use the folder you actually
generated. This runner currently supports MS-COCO only, not directory-format
ImageNet/CAMO/CelebA-HQ configs.

## 2. Validate inputs without a model

```bash
python src/scripts/explations/OD_yolo.py \
  --config MSCOCO_epito \
  --limit 2 \
  --dry-run
```

This prints selected image IDs, methods, partition paths, and output root. It
requires only Python/PyYAML, does not download weights or create results, and
checks that images exist and their IDs occur in the COCO annotations. Incomplete
partition outputs without their completed metadata/files are excluded.

## 3. Run a two-image experiment

```bash
python src/scripts/explations/OD_yolo.py \
  --config MSCOCO_epito \
  --limit 2 \
  --max-evals 50 \
  --eval-batch-size 16 \
  --auc-batch-size 8 \
  --verbose-level medium \
  --model yolo11s \
  --bg-type noise \
  --class-aggregation sum \
  --no-heatmaps
```

The default experiment runs exactly these methods, regardless of legacy
`evaluation.methods` in the config:

| Result key         | Explanation method             | Partition input               |
| ------------------ | ------------------------------ | ----------------------------- |
| `BPT`              | Stock BPT explanations         | Image-built hierarchy         |
| `SAM_c`            | BPT + SAM_c guidance           | `<image_id>_compact.npy`      |
| `SAM_c_kernelshap` | Kernel SHAP over SAM_c regions | Same `<image_id>_compact.npy` |

No raw-SAM, boosted, filled, or component-refined variant is run. Both SAM_c
methods use the same saved map and the same loader cap/sanitization. Generate
partitions with `--save-partitions compact` (or include compact among other stages)
before this experiment. The new script requires SAM_c maps for each selected
image and validates them before loading model weights.

`--max-evals` controls the BPT explanation budget. `--kernel-shap-samples` controls
the Kernel SHAP coalition budget (defaults to `--max-evals`). A subset of the
three methods can be explicitly selected with `--methods`, but other variants
and the old `auto` option are no longer accepted.

Increase or remove `--limit` for a larger run. `--image-ids 139 285` selects
specific completed COCO images. Other options from the original script remain
available through `--help`, including YOLO weights, background replacement,
Kernel SHAP budgets, plotting controls, and fail-fast behavior.

## Outputs and resume

Default outputs use the config's output root, then:

```text
OD_yolo/<mask_dir_final>/bg_<background>_B<max-evals>/
```

For example, black background and budget 500 save to:

```text
OD_yolo/MSCOCO_sam2_large_62_refined_200/bg_black_B500/139/000000000139_xai_results.png
```

There are no extra method, YOLO-model, or experiment-ID folders. The selected
model, methods, and experiment metadata remain recorded in `run_info.json`.
`--output-dir` overrides the output base; the `bg_<background>_B<max-evals>`
folder is still appended. Use separate output bases when comparing models or
other settings with the same background and budget.

Outputs include `run_info.json`, experiment mappings, partition/timing summaries,
updated per-image AUC CSVs, attribution images, aggregate plots, and
`xai_results_report.html`. Existing complete images are skipped by default;
`--no-resume` recomputes them. Use a distinct output directory for changes to
method sets, custom weights, or other settings you want to compare independently.
Recorded generation timings are limited to the stage timing data present in the
partition metadata; notebook batch stages may only have a combined stage time.

Local checks cover input/method selection, partial-output filtering, output-path
isolation, dry runs, and detector-score sum/max aggregation. Actual YOLO/SHAP GPU
execution must be tested on EPITO with the dataset and model weights.

## Show ground truth for the explained class

Add `--show_gt` (or `--show-gt`) to the same command:

```bash
python src/scripts/explations/OD_yolo.py \
  --config MSCOCO_epito \
  --limit 2 \
  --max-evals 50 \
  --kernel-shap-samples 50 \
  --show_gt
```

The class is the first class actually selected by the explainer, for example
chair. It is matched to COCO categories by name, not the YOLO class index (the
numbering differs). All segmentation instances of that class, including crowd
annotations, are decoded through COCO and unioned into a boolean mask. Other
classes are excluded. Polygon and RLE annotations use COCO's mask decoder.

Each image saves `<image_no>_gt.npy`, `<image_no>_gt.png` (mask + overlay), and
`<image_no>_gt.json` (class/category/annotation IDs and instance count). The XAI
grid adds a GT overlay, and the HTML report includes the standalone GT preview.
The predictions JSON also records the GT metadata. If that class has no matching
segmentation, an empty mask is saved with an explicit status; it is not replaced
by another class or a bounding box. Unknown custom detector classes are marked
as absent from COCO categories.

The standalone GT files are saved even with `--no-plots` or `--heatmaps-only`.
GT is for visualization; it does not alter explanations or the existing
faithfulness/AUC calculations. Runs with `--show_gt` require GT files before an
image is considered complete for resume, so older results without GT are
recomputed. No SAM or partition regeneration is required.
