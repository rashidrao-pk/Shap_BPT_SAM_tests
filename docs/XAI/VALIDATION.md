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
