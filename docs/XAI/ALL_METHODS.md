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

The existing 2×3 figure is retained. `<image-id>_all_methods.png` adds a wide
heatmap strip and, when GT has foreground, a second row of GT overlap panels.
The shared HTML report lets you scroll the wide comparison horizontally.
`--no-heatmaps` skips individual PNGs but keeps the comparison. `--no-plots`
keeps numeric maps, metrics, and method status without figures.

## Methods and budgets

The comparison includes the existing BPT, SAM_c and SAM_c_kernelshap methods,
plus:

| Method | Default setting | Meaning |
| --- | --- | --- |
| BPT-100/500/1000 | `--bpt-budgets 100 500 1000` | ShapBPT evaluation budgets |
| AA-100/500/1000 | Same budgets | Axis-aligned hierarchical explanations, as in the example |
| LIME-50/100/200 | `--lime-budgets 50 100 200` | Perturbation samples per background |
| LRP | Captum epsilon propagation | ResNet-50 logit relevance |
| GradCAM | Last CNN block / ViT pre-attention normalization | Positive class activation map |
| IDG | `--idg-steps 50` | Integrated **Decision** Gradients |
| GradExp | `--gradient-samples 20` | SHAP GradientExplainer expected gradients |
| GradShap | `--gradient-samples 20` | Captum GradientShap, noise standard deviation 0.0001 |

`--max-evals` still controls the original three methods; it does not replace the
explicit comparison budgets. LIME uses SLIC with approximately budget/5 regions
and averages across the configured background images. Its labels are sample
budgets, not region counts. This uses SLIC rather than the old notebook's tuned
quickshift segmentation, so its maps need not reproduce that notebook exactly.

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
