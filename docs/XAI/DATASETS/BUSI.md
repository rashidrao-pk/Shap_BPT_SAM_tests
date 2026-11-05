# BUSI explanations with all methods

One-time downloads: [BUSI setup](../Partitions/DATASETS/BUSI_SETUP.md). First [generate compact partitions](../Partitions/DATASETS/BUSI.md). The all-method workflow reuses those partitions and the frozen SigLIP checkpoint in `checkpoints/busi_siglip`.

## Same shared workflow as DETR and the other datasets

Set `data.mask_dir_final` in `BUSI_epito.yaml` to the desired SAM1 or SAM2 folder, then run:

```bash
python -u src/scripts/explanations/OC_BUSI.py \
  --config BUSI_epito --checkpoint checkpoints/busi_siglip \
  --show-gt --limit 1 --explain_all --budget 1000 \
  --no-heatmaps --auc-num-samples 200 --local-files-only --bg-type full
```

Remove `--limit 1` to explain all completed partition images. The three partition methods share one budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` also requests LIME-1000, Shap-AA-1000, LRP, GradCAM, IDG, GradExp, and GradShap. SigLIP GradCAM uses patch tokens without a CLS token; LRP remains explicitly unsupported and has no substitute attribution map. Optional methods require `requirements-xai-all.txt`.

The loader uses the checkpoint's label order and the BUSI model-card transforms: RGB, direct 224×224 resize, tensor conversion, and mean/std 0.5. No `preprocessor_config.json` is required. Masked predictions use the fixed original predicted class and its softmax probability. The model weights remain frozen.

## Keep the existing BUSI command

The standalone command now accepts `--explain_all` and routes to the same shared workflow:

```bash
python -u src/scripts/run_busi_shapbpt.py run \
  --config BUSI_epito --model-dir checkpoints/busi_siglip --limit 1 \
  --sam sam1 --regions 62 --budget 1000 --explain_all --batch-size 8 \
  --show-gt --no-heatmaps --auc-num-samples 200 --bg-type full
```

Use `--sam sam2` for SAM2 or `--sam sam1 sam2` for both. It prefers the configured folder when it matches the requested SAM model/region budget. Otherwise it looks for a single matching saved compact folder under `output.partitions`. If multiple folders match, select one explicitly:

```bash
python -u src/scripts/run_busi_shapbpt.py run \
  --config BUSI_epito --model-dir checkpoints/busi_siglip --limit 1 \
  --sam sam1 --regions 62 --budget 1000 --explain_all --batch-size 8 \
  --partition-folder BUSI_sam1_vit_b_62_compact-sam_50 \
  --show-gt --no-heatmaps --auc-num-samples 200
```

This mode reads existing compact partitions, so SAM checkpoint flags are unnecessary. To generate partitions first, use the dataset partition generator. The original command without `--explain_all` retains its separate hierarchical SAM/BPT pipeline and generation settings.

All-method mode creates config snapshots under `results/busi/all_methods_configs/` without changing your original config. Explanations and evaluations follow the source config's `output.explanations` and `output.evaluations`, under runner `OC_BUSI`, separated by partition folder/background/budget. `--bg-type` defaults to `full` in all-method mode, averaging black, gray, white, blurred and noise replacements.

## Plots, GT, AUC, and resume

`--show-gt` draws black lesion boundaries on attribution maps. Multiple mask files are united without flipping; empty annotated masks have no contour, and missing masks are reported as unavailable. The lesion region is not evidence that the predicted benign/malignant class is correct.

The shared runner saves the normal combined XAI grid, an all-method comparison, numeric maps, and AUC/evaluation outputs. `--no-heatmaps` skips individual method PNGs while retaining the combined plots. AUC requests 200 points by default; repeated masks can reduce the actual count. GT diagnostics include tie-aware pointing accuracy and GT-selected best IoU; best IoU is not a fixed-threshold test score.

Timing is reported per image. Repeat the same command to resume completed images; an interrupted image repeats its methods. Keep options unchanged and omit `--no-resume` for continuation.
