# ImageNet explanations

Run from the repository root on a GPU node. First [download the dataset](../../DATASETS/ImageNet.md), get the checkpoint ([ResNet-50](../../MODELS/RESNET50.md), [VGG16](../../MODELS/VGG16.md), [ViT-B/16](../../MODELS/VIT_B16.md)), and [generate partitions](../../Partitions/DATASETS/ImageNet.md). Make sure `data.mask_dir_final` names the generated folder.

## Run

Use `--gt-only` so that `--limit` counts only images with an ImageNet-S mask:

```bash
python src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only --limit 2 \
  --budget 1000 --no-heatmaps --explain_all
```

Replace `OC_ResNet.py` with `OC_VGG.py` or `OC_ViT.py` for the other classifiers.

The three partition methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` adds LIME, Shap-AA, LRP, GradCAM, IDG, GradExp, and GradShap, and needs `requirements-xai-all.txt`. Remove `--limit 2` for all completed partitions. Add `--dry-run` to check paths without loading the model.

All three use torchvision's default ImageNet weights and preprocessing, and explain the top prediction, fixed for each image.

## Ground truth

`--show-gt` shows the ImageNet-S mask for the **predicted** class, through classifier index → synset → ImageNet-S label. If the prediction is not the annotated class, the mask is empty and the status says so.

```bash
--gt-class foreground   # Every annotated object
--gt-class n07930864    # One synset
```

A numeric `--gt-class` is an ImageNet-S label ID, not a classifier index. See [GT formats and mapping](../GROUND_TRUTH.md) for details, including "annotation not available".

## Reading the results

- ImageNetS50 covers 752 of the 50,000 validation images. Without `--gt-only`, most selected images have no mask.
- These runs measure faithfulness to the model, not label accuracy.

## Outputs and resume

```text
<output.explanations>/OC_ResNet/<partition-folder>/bg_full_B1000/<image-id>/
<output.evaluations>/OC_ResNet/<partition-folder>/bg_full_B1000/
```

Repeat the same command to resume: completed images are skipped. `--no-heatmaps` skips individual method images, and keeps the combined plots and numeric results.
