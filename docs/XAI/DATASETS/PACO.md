# PACO explanations

Run from the repository root on a GPU node. First [download the dataset](../../SETUP/DATASETS/PACO.md), get the checkpoint ([ResNet-50](../../SETUP/MODELS/RESNET50.md)), and [generate partitions](../../Partitions/DATASETS/PACO.md). Make sure `data.mask_dir_final` names the generated folder.

## Run

```bash
python src/scripts/explanations/OC_PACO.py \
  --config PACO_epito \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --explain_all
```

### SAM version and compact partitions

The same command for each SAM model (`--sam`, its folder from the generator) and each saved compact file (`--compact-strategy`). Without `--compact-strategy`, the config's default `split` is used. Results of `split` go to `<partition-folder>_split`; each run prints `Partitions: ... | compact file: ...` first.

```bash
# SAM-1 ViT-B, greedy compact partitions
python src/scripts/explanations/OC_PACO.py \
  --config PACO_epito --sam sam1 --compact-strategy greedy \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --explain_all

# SAM-1 ViT-B, split compact partitions
python src/scripts/explanations/OC_PACO.py \
  --config PACO_epito --sam sam1 --compact-strategy split \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --explain_all

# SAM-2 Large, greedy compact partitions
python src/scripts/explanations/OC_PACO.py \
  --config PACO_epito --sam sam2 --compact-strategy greedy \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --explain_all

# SAM-2 Large, split compact partitions
python src/scripts/explanations/OC_PACO.py \
  --config PACO_epito --sam sam2 --compact-strategy split \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --explain_all
```

The three partition methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` adds Shap-AA and LIME; `--gradient_based` next to it also adds LRP, GradCAM, IDG, GradExp and GradShap, computed and plotted last. Both need `requirements-xai-all.txt`. Remove `--limit 2` for all completed partitions. Add `--dry-run` to check paths without loading the model.

The classifier is a frozen ImageNet ResNet-50, and the explained class is its top prediction, fixed for each image. `data.annotation_file` and `ground_truth.annotation_file` must select the same split, or the runner stops.

## Ground truth

`--show-gt` outlines all annotated parts in the image, leaving out whole-object masks. To select a category, use its exact name from the annotation JSON:

```bash
--gt-class "mug:handle"   # One part
--gt-class mug            # Whole objects
```

## Reading the results

- ImageNet classes are not PACO's 75 objects, so the explained class can differ from the annotated object.
- Images average 4.6 annotated objects, and the default contour covers all of them. Only 789 of 2,410 val images have one annotated object, and these give the clearest localization.
- Not every object is annotated, so attribution outside the contour is not automatically wrong.

## Outputs and resume

```text
<output.explanations>/OC_PACO/<partition-folder>/bg_full_B1000/<image-id>/
<output.evaluations>/OC_PACO/<partition-folder>/bg_full_B1000/
```

Repeat the same command to resume: completed images are skipped. Individual method images are not saved by default (`--heatmaps` adds them; `--no-heatmaps` is accepted and changes nothing); the combined plots and numeric results are always kept. With `--show-gt`, the separate GT figure `<id>_gt.png` is saved only with `--gt-png`.
