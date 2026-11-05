# Compare AUC with 100 and 200 points

The gain to measure is **evaluation precision versus runtime**, not improved explanations or increased model accuracy. Explanations stay fixed. A larger AUC value by itself does not indicate a better resolution choice.

## Run on saved explanations

Use an existing shared-runner experiment folder containing `run_info.json`, per-image predictions, and `*_attributions.npz`. YOLO numeric maps are saved by its `--explain_all` mode. The comparison loads the original model/settings and reuses saved maps, with the original fixed class and background. It never recomputes BPT, Kernel SHAP, or gradient explanations.

For example, inside an allocated GPU session:

```bash
python -u src/scripts/explanations/compare_auc_points.py \
  --run-dir results/explanations/OD_DETR/MSCOCO_sam1_vit_b_62_compact-sam_200/bg_black_B1000 \
  --limit 3 \
  --methods BPT-1000 BPT+SAM_c-1000 SAM_c+KernelSHAP-1000 \
  --points 100 200 --reference-points 1000 --repeats 3 \
  --tolerance 0.005 --device cuda
```

Replace `--run-dir` with the actual completed experiment directory. Run the same comparison for YOLO and each SAM variant, or a classifier such as ResNet/ViT/SigLIP. Methods are exact saved NPZ keys; omit `--methods` to include every saved map. To compare region maps against dense gradient maps, include the gradient methods that are actually saved for that classifier. Unsupported methods have no numeric map to compare.

`--limit` selects the first N saved map archives; `--image-ids` optionally restricts the pool using exact saved IDs. Model paths are restored from the experiment's metadata. Use `--checkpoint` if the same weights have moved. The original image files must remain accessible. Full/background scores are checked against the saved predictions; mismatches stop the comparison rather than mixing model/scoring setups.

## What is measured

For each image, saved method, and insertion/deletion direction:

- Recompute 100-point and 200-point curves using the same saliency map, model, target, and baseline.
- Evaluate a denser 1000-point reference. This is an approximation, not exact AUC ground truth.
- Report absolute normalized-AUC change and each resolution's error against that reference.
- Report actual distinct masks, background-expanded model-image counts, and median runtime across repeats.
- Check whether rankings of the saved methods change between resolutions.

Model loading, explanation generation, endpoint warm-up, and reference computation are excluded from the measured candidate runtimes. GPU synchronization surrounds timing; candidate order alternates across repeats. Candidate measurements include mask/threshold construction, preprocessing, model inference, and integration. Reference evaluation adds benchmark cost, even though it is excluded from the candidate timing ratio.

Duplicate masks are skipped by the current AUC routine. Therefore 200 requested points need not produce 200 distinct evaluations. A map with approximately 63 uniform regions may use the same distinct masks at both settings; a dense map can approach twice as many model evaluations at 200. Threshold-computation overhead can still increase even when the mask count does not.

## Read the comparison and conclusion

Outputs are written under the source experiment's saved evaluation directory in `auc_point_comparison/`. If that directory is absent from old metadata, the output is placed under the source run instead. Override with `--output-dir` if desired.

- `auc_resolution_comparison.png`: evaluated points, median runtime, and worst reference error by method.
- `measurements.csv`: every resolution/image/method/direction measurement.
- `paired_comparisons.csv`: paired AUC changes, reference errors, runtime ratios, and mask counts.
- `curves/`: the full candidate and reference curves.
- `summary.json` and `conclusion.md`: measured gain and a scope-limited recommendation.

The predeclared default tolerance is **0.005 absolute normalized AUC**, configurable before running. The conclusion favors 100 if its worst reference error and the 100/200 changes meet tolerance without ranking changes. It favors 200 if only 200 meets the reference tolerance. Unstable repeated scores, unresolved reference errors, or other mixed outcomes produce no blanket recommendation.

Mask counts in the paired summary describe one evaluation pass per resolution; they exclude extra timing repeats, endpoint checks, and reference evaluations. The reference need not improve monotonically with every requested point count because quantile grids and tied values differ. Avoid treating a small pilot as a universal recommendation. `comparison_status.json` must say `complete` before interpreting a report; a failed rerun clears the previous conclusion. The benchmark does not resume individual comparisons; start with a few images. Repeating it replaces its own reports and leaves the source explanations unchanged.

## Controlled illustration (not MS-COCO evidence)

A local check used a 20×20 saliency map and synthetic score `(retained_pixel_fraction)**2`, with a 1000-point reference:

| Map | Actual masks at 100 / 200 | AUC at 100 / 200 | Absolute reference error at 100 / 200 |
| --- | --- | --- | --- |
| Dense, 400 distinct saliency values | 100 / 200 | 0.33335047 / 0.33333758 | 0.00001609 / 0.00000320 |
| 63 nearly equal-size tied regions | 64 / 64 | 0.33337605 / 0.33337605 | 0 / 0 |

The dense-map error decreased about fivefold, but the absolute AUC change was only about 0.000013. The tied-region example gained no AUC change or additional model evaluations. These checks demonstrate why the saved-map benchmark is needed; they do not establish the gain for the real models/datasets.

Keep the current 200-point experiment default until your paired comparison supports a different choice. If your budget-selection pilot validated a budget at 200 points, its selection file requires that AUC resolution; a resolution benchmark does not automatically revalidate that budget file at 100 points.
