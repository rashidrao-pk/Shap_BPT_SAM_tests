# YOLO explanations from MS-COCO partitions

[OD_yolo.py](../../src/scripts/explations/OD_yolo.py) is a separate copy of
`run_yolo_full_all.py`. The original is unchanged. It retains YOLO prediction,
ShapBPT/region Kernel SHAP, deletion/insertion AUC, faithfulness metrics,
per-image heatmaps, plots, CSV summaries, resume checks, and the HTML report.

Use the same EPITO environment as [EPITO.md](EPITO.md#1-start-a-session).
No SAM inference or partition generation is run by this script.

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
  --methods auto \
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
  --methods auto \
  --limit 2 \
  --max-evals 50 \
  --eval-batch-size 16 \
  --auc-batch-size 8 \
  --verbose-level medium \
  --model yolo11s \
  --bg-type noise \
  --class-aggregation sum
```

Method selection:

- Omit `--methods` to use `evaluation.methods` from the config.
- `--methods auto` uses saved stages common to all selected images, plus `BPT` and `sam_kernelshap` when SAM maps exist.
- `--methods BPT sam compact refined` explicitly selects methods; unavailable stages fail before model loading.

The modern stage names are `sam`, `compact`, `boosted`, `filled`, `boosted_filled`,
and `refined`. Old `greedy_selection`/`compact_selection` filenames are not
silently substituted for different algorithms. All loaded partition maps still
go through the original XAI loader's 63-label cap/sanitization.

Increase or remove `--limit` for a larger run. `--image-ids 139 285` selects
specific completed COCO images. Other options from the original script remain
available through `--help`, including YOLO weights, background replacement,
Kernel SHAP budgets, plotting controls, and fail-fast behavior.

## Outputs and resume

Default outputs use the config's output root, then:

```text
OD_yolo/<mask_dir_final>/<yolo-model>/xai_results_E1_1_1/
```

The experiment suffix above corresponds to noise background and `max_evals=50`.
The runner prints the actual final results path. Each partition dataset and YOLO
model has a separate output namespace. `--output-dir` explicitly overrides that
namespace, as in the original runner.

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
