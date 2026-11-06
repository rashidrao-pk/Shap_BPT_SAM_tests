# PACO explanations

Run from the repository root on a GPU node. First [download the dataset](../../DATASETS/PACO.md), get the checkpoint ([ResNet-50](../../MODELS/RESNET50.md)), and [generate partitions](../../Partitions/DATASETS/PACO.md). Make sure `data.mask_dir_final` names the generated folder.

## Run

```bash
python src/scripts/explanations/OC_PACO.py \
  --config PACO_epito \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --explain_all
```

The three partition methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` adds LIME, Shap-AA, LRP, GradCAM, IDG, GradExp, and GradShap, and needs `requirements-xai-all.txt`. Remove `--limit 2` for all completed partitions. Add `--dry-run` to check paths without loading the model.

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

Repeat the same command to resume: completed images are skipped. `--no-heatmaps` skips individual method images, and keeps the combined plots and numeric results.
