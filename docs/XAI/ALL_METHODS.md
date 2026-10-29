# Compare all explanation methods

Add `--explain_all` (or `--explain-all`) to a runner. The default three-method
experiment stays unchanged when this flag is absent.

```bash
python src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only --limit 2 \
  --explain_all --no-heatmaps --no-resume
```

Optional dependencies, inside the existing SAM/PyTorch environment:

```bash
python -m pip install -r requirements-xai-all.txt
```

The requirements pin NumPy 1.26.4, Captum 0.8.0, SHAP 0.49.1 and OpenCV
4.11.0.86 for the SAM environment. Earlier unbounded requirements could upgrade
NumPy to 2.x and conflict with SAM3. After syncing the updated requirements,
repair that installation with:

```bash
python -m pip install -r requirements-xai-all.txt
python -m pip check
```

If `pip check` reports **Transformers 5.15.0** with `huggingface-hub 2.x`, its
reported requirement is `huggingface-hub>=1.5,<2`. Repair that separate conflict:

```bash
python -m pip install 'huggingface-hub>=1.5,<2'
python -m pip check
```

Do not force-reinstall the shared PyTorch packages. Verify imports before running
an experiment:

```bash
python -c "import numpy, cv2, torch, torchvision, captum, shap; print(numpy.__version__, cv2.__version__, torch.__version__, torchvision.__version__, captum.__version__, shap.__version__)"
```

The 2×3 figure shows the three partition-based methods at the highest budget. `<image-id>_all_methods.png` adds a wide
heatmap strip and, when GT has foreground, a second row of GT overlap panels.
There is no standalone GT column; GT appears as boundaries on the attribution maps.
The shared HTML report lets you scroll the wide comparison horizontally.
`--no-heatmaps` skips individual PNGs but keeps the comparison. `--no-plots`
keeps numeric maps, metrics, and method status without figures.

## Methods and budgets

In `--explain_all` mode the complete method list is:

| Method | Default setting | Meaning |
| --- | --- | --- |
| BPT-500/1000 | `--partition-budgets 500 1000` | BPT evaluation budgets |
| BPT+SAM_c-500/1000 | Same budgets | BPT initialized with saved SAM_c regions |
| SAM_c+KernelSHAP-500/1000 | Same budgets | Kernel SHAP coalition samples on saved SAM_c regions |
| LIME-1000 | `--lime-budgets 1000` | 1000 perturbation samples per background |
| Shap-AA-1000 | `--aa-budget 1000` | Axis-aligned hierarchical SHAP |
| LRP | Captum epsilon propagation | ResNet-50 logit relevance |
| GradCAM | Last CNN block / ViT pre-attention normalization | Positive class activation map |
| IDG | `--idg-steps 50` | Integrated **Decision** Gradients |
| GradExp | `--gradient-samples 20` | SHAP GradientExplainer expected gradients |
| GradShap | `--gradient-samples 20` | Captum GradientShap, noise standard deviation 0.0001 |

`--partition-budgets` (alias `--bpt-budgets`) controls all three partition-based
methods. In all-method mode `--max-evals` and `--kernel-shap-samples` do not
replace those explicit budgets. No extra unbudgeted copies are computed. The
compact 2×3 plot uses the highest partition budget; the wide figure shows all
methods. Without `--explain_all`, the original three-method behavior is unchanged.

LIME requests **250 SLIC segments for MS-COCO** and **100 for every other dataset**.
Use `--lime-segments N` to override. SLIC is approximate; the requested and actual
region counts are saved in method status. LIME sample budgets are independent of
segment counts. It averages across the configured background images. The original
notebook used tuned quickshift, so its maps need not reproduce that notebook exactly.

Gradient methods use the classifier's actual weights and preprocessing. Input
attributions are summed across color channels without taking absolute values or
rescaling them to force completeness. Maps are restored through the inverse
resize/center crop; pixels outside the model's crop receive zero attribution.
LRP uses logits; other gradient methods use classification probabilities. These
choices are recorded in method metadata. GradExp and GradShap use configured
background images as their baseline distribution. They have different sampling
implementations; IDG is not ordinary Integrated Gradients under a different name.

IDG follows adaptive sampling from decision-score slopes, then weights path
input gradients by those slopes and interval widths. The implementation handles
constant-score paths, partial batches, and preserves positions of empty sampling
intervals. It does not copy the upstream empty-interval indexing issue.

## Architecture support

- **OC_CelebA:** all 13 methods with the E5 CelebModel checkpoint; raw attribute
  logits, direct resize, and GradCAM at `unit8.conv`. See [CelebA setup](CELEBA.md).
- **OC_ResNet:** all listed methods, including LRP with explicit residual addition
  modules on a separate model copy. The copy's logits are checked against the
  scoring model before attribution.
- **OC_ViT / COD_ViT:** BPT/AA/LIME, GradCAM, IDG, GradExp and GradShap. LRP is
  marked **unsupported**: torchvision ViT requires a transformer-specific relevance
  backend. No fake zero map is saved as an explanation.
- **OD_DETR / OD_yolo:** BPT/AA/LIME comparisons; gradient columns explicitly report
  unsupported because detector scoring does not yet expose a gradient adapter.

Each requested method has a status in `<image-id>_method_status.json`. Unsupported
methods are labelled in the figure and excluded from numeric metrics. Unexpected
errors fail the image and appear in `failures.json`. YOLO comparisons use a separate completion marker with a settings signature and
artifact checks. The shared runners use their existing completion manifests.

## GT IoU row

Use `--show-gt`, and for ImageNet optionally `--gt-only`. The heatmap row has GT
contours. The second row uses **blue = true positive, orange = false positive,
magenta = false negative, white = true negative**.

IoU is the maximum over distinct heatmap thresholds on that image, with ties kept
together and ImageNetS ignore pixels excluded. This is an **oracle threshold
selected using GT**, useful for comparison, not an unbiased localization estimate.
The selected threshold and best IoU are recorded alongside method timings and
AUC/faithfulness metrics. Missing or empty selected-class GT has no IoU score.

The per-image NPZ contains every successful method's attribution. The shared
runner includes all successful methods in CSV summaries and the HTML report.
Use `--no-resume` when adding this mode to an existing standard experiment, or
choose `--output-dir` for a separate shared-runner experiment.

For a quick check use small budgets and `--no-auc`, then increase them:

```bash
python src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only --limit 1 \
  --explain_all --bpt-budgets 100 --lime-budgets 50 \
  --gradient-samples 10 --idg-steps 20 \
  --attribution-batch-size 2 --no-auc --no-heatmaps --no-resume
```

## References

- [Original experiment notebook](https://github.com/rashidrao-pk/shap_bpt_tests/blob/main/notebooks/E1_E2_E3_E7/N1_1_Run_experiments_testing.ipynb)
- [LIME image implementation](https://github.com/marcotcr/lime/blob/master/lime/lime_image.py)
- [Captum LRP](https://captum.ai/api/lrp.html), [GradientShap](https://captum.ai/api/gradient_shap.html)
- [Integrated Decision Gradients reference](https://github.com/chasewalker26/Integrated-Decision-Gradients/blob/main/util/attribution_methods/saliencyMethods.py)

Local checks use synthetic annotations/partitions and CPU pretrained ResNet/ViT
smoke runs. They do not establish localization accuracy on the EPITO datasets.

## Notebook heatmap styling

Attribution figures use `shap_bpt.shapley_values_colormap`, the notebook's custom
blue–white–red map, with `vmax = quantile(abs(map), 0.99)` over all pixels and
`vmin = -vmax`, independently for each method. A zero limit uses a small positive
fallback. This affects display only; saved attribution values and IoU are unchanged.
GT panels use `binary` (black foreground on white); attribution panels use black
`mark_boundaries(..., mode="thick")` overlays, matching
`fun_plot_heatmaps_filtered` in the reference notebook.

GT boundaries also have a black vector outline with a thin white halo so they
remain visible when high-resolution masks are downsampled into narrow comparison
panels. Line width is set in display points, not source-image pixels. GT arrays,
attribution values and IoU calculations are unchanged.
