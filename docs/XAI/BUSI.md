# BUSI explanations

One-time downloads: [BUSI setup](../Partitions/DATASETS/BUSI_SETUP.md).

First [generate partitions](../Partitions/DATASETS/BUSI.md) and [download SigLIP](../Partitions/MODELS/SigLIP_BUSI.md). Ensure `data.mask_dir_final` matches your generated folder.

```bash
python src/scripts/explanations/OC_BUSI.py \
  --config BUSI_epito --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --local-files-only
```

Remove `--limit 2` for all completed partition images. The three methods are BPT, BPT+SAM_c, and SAM_c+KernelSHAP, sharing one budget. The target is the full input's predicted class and remains fixed for all masked predictions. AUC uses 200 requested points by default (`--auc-num-samples` overrides it).

`--show-gt` draws black lesion boundaries on attribution maps. Lesion masks indicate the annotated lesion region, not evidence that the predicted benign/malignant class is correct. Empty annotated masks have no contour; missing masks are reported as unavailable. Multiple lesion masks are united without flipping them.

Add `--explain_all` for optional baselines. SigLIP LRP and GradCAM currently report unsupported; no substitute map is produced. Other optional methods require the dependencies in `requirements-xai-all.txt`.

Explanations and evaluations follow `output.explanations` and `output.evaluations`; partition generation remains separate. Each run reports per-image timing. With GT, metrics include tie-aware pointing accuracy and GT-selected best IoU. Best IoU is a diagnostic rather than a fixed-threshold test score; empty lesion masks have no pointing score.
