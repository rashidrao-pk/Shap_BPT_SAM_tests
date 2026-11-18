# Kvasir-SEG explanations

Run from the repository root on a GPU node. First [download the dataset](../../DATASETS/KVASIR_SEG.md), get the checkpoint ([Kvasir v2 ViT](../../MODELS/KVASIR_VIT.md)), and [generate partitions](../../Partitions/DATASETS/KVASIR_SEG.md). Make sure `data.mask_dir_final` names the generated folder.

## Run

```bash
python src/scripts/explanations/OC_KvasirSEG.py \
  --config KvasirSEG_epito --target-class 6 --show-gt --limit 1 \
  --budget 1000 --no-heatmaps --explain_all
```

### SAM version and compact partitions

The same command for each SAM model (`--sam`, its folder from the generator) and each saved compact file (`--compact-strategy`). Without `--compact-strategy`, the config's default `split` is used. Results of `split` go to `<partition-folder>_split`; each run prints `Partitions: ... | compact file: ...` first.

```bash
# SAM-1 ViT-B, greedy compact partitions
python src/scripts/explanations/OC_KvasirSEG.py \
  --config KvasirSEG_epito --sam sam1 --compact-strategy greedy --target-class 6 --show-gt --limit 1 \
  --budget 1000 --no-heatmaps --explain_all

# SAM-1 ViT-B, split compact partitions
python src/scripts/explanations/OC_KvasirSEG.py \
  --config KvasirSEG_epito --sam sam1 --compact-strategy split --target-class 6 --show-gt --limit 1 \
  --budget 1000 --no-heatmaps --explain_all

# SAM-2 Large, greedy compact partitions
python src/scripts/explanations/OC_KvasirSEG.py \
  --config KvasirSEG_epito --sam sam2 --compact-strategy greedy --target-class 6 --show-gt --limit 1 \
  --budget 1000 --no-heatmaps --explain_all

# SAM-2 Large, split compact partitions
python src/scripts/explanations/OC_KvasirSEG.py \
  --config KvasirSEG_epito --sam sam2 --compact-strategy split --target-class 6 --show-gt --limit 1 \
  --budget 1000 --no-heatmaps --explain_all

# MedSAM, greedy compact partitions
python src/scripts/explanations/OC_KvasirSEG.py \
  --config KvasirSEG_epito --sam medsam --compact-strategy greedy --target-class 6 --show-gt --limit 1 \
  --budget 1000 --no-heatmaps --explain_all

# MedicoSAM, greedy compact partitions
python src/scripts/explanations/OC_KvasirSEG.py \
  --config KvasirSEG_epito --sam medicosam --compact-strategy greedy --target-class 6 --show-gt --limit 1 \
  --budget 1000 --no-heatmaps --explain_all
```

The three partition methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` adds LIME, Shap-AA, LRP, GradCAM, IDG, GradExp, and GradShap, and needs `requirements-xai-all.txt`. Remove `--limit 2` for all completed partitions. Add `--dry-run` to check paths without loading the model.

The checkpoint path comes from `model.checkpoint` in the config. `--target-class 6` fixes the explained class to `polyps`, so every map answers the same question as the mask. Without it, the runner explains the top prediction, which can be another class.

LRP is reported as `unsupported` for this ViT, with no substitute map. All other methods run.

## Ground truth

`--show-gt` outlines the expert polyp mask. Results also include pointing accuracy and best IoU. Best IoU picks the best threshold using GT, so it is a diagnostic, not a test score.

## Reading the results

- Assume the model saw these images in training: Kvasir-SEG comes from Kvasir v2's polyp class. Maps describe this model, but do not report them as held-out performance.
- Black borders cover about 15% of the average image. Keep the default `--bg-type full`, because `black` alone can resemble the border.
- Research diagnostics only, not clinical evidence.

## Outputs and resume

```text
<output.explanations>/OC_KvasirSEG/<partition-folder>/bg_full_B1000/<image-id>/
<output.evaluations>/OC_KvasirSEG/<partition-folder>/bg_full_B1000/
```

Repeat the same command to resume: completed images are skipped. `--no-heatmaps` skips individual method images, and keeps the combined plots and numeric results.
