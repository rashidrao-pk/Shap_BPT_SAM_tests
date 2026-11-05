# PartImageNet explanations

First [download assets and configure a split](../Partitions/DATASETS/PARTIMAGENET_SETUP.md), then [generate partitions](../Partitions/DATASETS/PARTNET.md). Ensure `data.mask_dir_final` matches the generated folder.

Inside an allocated GPU node:

```bash
python src/scripts/explanations/OC_PartImageNet.py \
  --config PartImageNet_epito \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps
```

Remove `--limit 2` for all completed partitions in the configured split. Three methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. Add `--explain_all` for LIME, Shap-AA, and the optional gradient methods. Use the existing optional-method dependency setup.

The frozen ImageNet classifier's top predicted class is fixed throughout each image's explanation. No classifier fitting or part-supervision training is performed. The checkpoint is local, so loading it does not require downloading another backbone.

`--show-gt` draws the union of annotated parts as a foreground contour on attribution maps. To show one part, add its exact category name in quotes or numeric category ID:

```bash
--gt-class "Quadruped Head"
```

Read the selected JSON's `categories` list for available names. Part categories differ from ImageNet object classes: foreground contours do not imply that the classifier's prediction matches the annotated object. Individual part labels and annotation IDs are saved in the GT artifacts; the default contour outlines their union.

The normal 2×3 plot shows input/BPT/SAM partitions followed by the three attribution maps, with no standalone GT column. Insertion/deletion AUC requests 200 points by default; override with `--auc-num-samples`. `--no-heatmaps` skips individual maps while retaining combined plots and numeric results. Timing and resume use the shared runner.

Paths come from `output.explanations` and `output.evaluations`, with runner `OC_PartImageNet`. Use a separate config/partition root when comparing dataset splits.
