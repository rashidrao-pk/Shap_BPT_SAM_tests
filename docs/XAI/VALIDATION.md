# Explanation runner validation

Automated checks:

```bash
python -m unittest discover -s tests -p test_model_explanations.py
```

Tests cover the four script defaults, dataset-specific image-ID handling,
lightweight dry runs, missing compact maps, background averaging, DETR sparse
category IDs and sum/max aggregation, custom ViT classifier head/label metadata,
real ShapBPT/Kernel SHAP execution, metric CSVs, plots, report links, and resume.
The end-to-end unit test uses an inexpensive deterministic classifier so it does
not require downloading pretrained weights.

CPU pretrained-model smoke checks use a bundled scikit-image coffee photo,
resized to 48 × 48, with four-quadrant label maps and small explanation budgets.
These are execution fixtures, not original MS-COCO/ImageNet/CAMO data or SAM maps.
The local environment had cached ResNet-50, ViT-B/16, and DETR weights; no new
weights were downloaded. DETR smoke checks use a 128-pixel processor resize.

Artifacts are in `results/xai_runner_smoke/`, with an HTML report for each runner
under `evaluations/`. The CAMO entry point is tested with the default ImageNet
classifier. A real CAMO-finetuned checkpoint and full EPITO GPU runs remain to be
validated with the user's data.

The four new runners use the shared `saliency_to_auc` and faithfulness functions.
They evaluate insertion and deletion once at `--auc-num-samples`; they do not
silently add fixed 500/2000-sample sweeps. `SAM_c_kernelshap` uses the same capped
compact labels as BPT+SAM_c. One-region partitions use the exact single-feature
contribution; multi-region maps use the existing weighted Kernel SHAP solver.

## All-method checks across datasets

Updated all-method routing was exercised with cached pretrained weights on CPU:

| Dataset runner    | Completed | Explicitly unsupported    |
| ----------------- | --------- | ------------------------- |
| ImageNet / OC_ViT | 12 of 13  | LRP                       |
| CAMO / COD_ViT    | 12 of 13  | LRP                       |
| MS-COCO / OD_DETR | 8 of 13   | All five gradient methods |
| MS-COCO / OD_yolo | 8 of 13   | All five gradient methods |

Every case had a positive target score, nonzero finite attribution maps, matched
synthetic GT, valid combined PNGs and an HTML report. Checked the exact method
list, both budgets for all three partition-based methods, requested LIME segment
counts (250 for COCO, 100 otherwise), numeric R², and saved arrays. SLIC returned
194 regions for the larger COCO fixture and 88 for the classifier fixtures.
CelebA-HQ's partition/GT preview was checked separately. The subsequently added
`OC_CelebA.py` runner and published-checkpoint smoke test are documented in
[CELEBA.md](CELEBA.md).

Artifacts and a report index are in `results/all_methods_dataset_smoke/`, with
`validation_summary.json` recording each check. The smoke runs used partition
budgets 8 and 12, LIME 8 samples, AA budget 8, two gradient samples and four IDG
steps, with AUC disabled. They validate routing/execution, not the cost or quality
of the production 500/1000 budgets. The images, annotations and SAM partitions
are synthetic fixtures, not the actual EPITO datasets. The default method matrix
is also checked by `tests/test_all_methods.py`.

The detector checks use a larger bundled coffee photo with nonzero detections;
the original 48-pixel fixture had zero detector scores and was not accepted as
sufficient evidence of working detector attributions. The CAMO default model
is still an ImageNet classifier, not a camouflage-trained segmentation model.

## First checks on EPITO data

Run one image with the requested default 500/1000 budgets. These commands omit
AUC for the initial check; remove `--no-auc` for a complete evaluation.

```bash
python src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only --limit 1 \
  --explain_all --no-heatmaps --no-resume

python src/scripts/explanations/OC_ViT.py \
  --config ImageNet_epito --show-gt --gt-only --limit 1 \
  --explain_all --no-heatmaps --no-resume

python src/scripts/explanations/COD_ViT.py \
  --config CAMO_epito --show-gt --limit 1 \
  --explain_all --no-heatmaps --no-resume

python src/scripts/explanations/OD_DETR.py \
  --config MSCOCO_epito --show-gt --limit 1 \
  --explain_all --no-heatmaps --no-resume

python src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito --show-gt --limit 1 \
  --explain_all --no-heatmaps --no-resume

python src/scripts/explanations/OC_CelebA.py \
  --config CelebAHQ_epito \
  --attribute Brown_Hair \
  --show-gt --limit 1 \
  --explain_all --no-heatmaps --no-resume

```

Unsupported methods remain explicitly labelled; their absence is not a failed
installation. Use the saved method-status JSON to check requested/actual LIME
regions and method budgets, and inspect the GT boundaries in the combined plots.
