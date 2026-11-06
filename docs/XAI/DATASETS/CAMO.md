# CAMO explanations

Run from the repository root on a GPU node. First [download the dataset](../../DATASETS/CAMO.md), get the checkpoint ([ViT-B/16](../../MODELS/VIT_B16.md)), and [generate partitions](../../Partitions/DATASETS/CAMO_EPITO.md). Make sure `data.mask_dir_final` names the generated folder.

## Run

```bash
python src/scripts/explanations/COD_ViT.py \
  --config CAMO_epito --show-gt --limit 1 \
  --budget 1000 --no-heatmaps --explain_all
```

The three partition methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` adds LIME, Shap-AA, LRP, GradCAM, IDG, GradExp, and GradShap, and needs `requirements-xai-all.txt`. Remove `--limit 2` for all completed partitions. Add `--dry-run` to check paths without loading the model.

The default classifier is torchvision's ImageNet ViT-B/16, and the explained class is its top ImageNet prediction, fixed for each image.

For a classifier trained on CAMO, add `--checkpoint /path/to/model.pt --class-names /path/to/classes.json`. It must match torchvision's ViT-B/16 classifier. Segmentation checkpoints need a different adapter.

## Ground truth

`--show-gt` shows the binary camouflaged-object mask. It does not depend on the predicted class.

## Reading the results

- ImageNet has no "camouflaged object" class, so the prediction may name the animal, the background, or something unrelated. Check the explained class before reading the mask overlap as success.
- Because the object is hidden by design, SAM can miss it. Check partitions with `view_partitions.py --show-gt`: if a region crosses the object edge, no explanation built on it can follow that edge.

## Outputs and resume

```text
<output.explanations>/COD_ViT/<partition-folder>/bg_full_B1000/<image-id>/
<output.evaluations>/COD_ViT/<partition-folder>/bg_full_B1000/
```

Repeat the same command to resume: completed images are skipped. `--no-heatmaps` skips individual method images, and keeps the combined plots and numeric results.
