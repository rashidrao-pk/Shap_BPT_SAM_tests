# CelebA-HQ attribute explanations on EPITO

This runner uses the custom **CelebModel** and checkpoint from
[the E5 notebook](https://github.com/rashidrao-pk/shap_bpt_tests/blob/main/notebooks/E5_CelebA/N1_Run_experiments_CelebA.ipynb).
It predicts 40 independent facial attributes. XAI explains the raw attribute
**logit**, matching the notebook, rather than applying ImageNet softmax.

## Download once

From the repository root, inside `pt_312_sam`:

```bash
python src/scripts/download_celeba_checkpoint.py
```

The downloader saves `src/checkpoints/celeba_model.pth`, checks its SHA-256, and
reuses an existing matching file. The source is the
[checkpoint included in the reference repository](https://github.com/rashidrao-pk/shap_bpt_tests/blob/main/notebooks/E5_CelebA/models/model.pth).
It is about 42 MB. No Kaggle login or retraining is required for this copy.
`CelebAHQ_epito.yaml` already points to this checkpoint. The checkpoint is not
committed to this repository: download it separately on EPITO.

## Run explanations

The configured partition folder must contain completed `<id>_compact.npy` maps
and `<id>_results.json`, just like the other dataset runners.

```bash
python src/scripts/explanations/OC_CelebA.py \
  --config CelebAHQ_epito \
  --attribute Brown_Hair \
  --show-gt --limit 1 \
  --explain_all --no-heatmaps --no-auc --no-resume
```

This runs BPT, BPT+SAM_c and SAM_c+KernelSHAP at 500 and 1000; LIME-1000 with
100 requested SLIC segments; Shap-AA-1000; and all five gradient baselines.
**LRP and GradCAM are supported for this CNN.** The GradCAM target is `unit8.conv`,
as in the notebook. The default background is gray. Remove `--no-auc` to add
faithfulness evaluation, and remove `--limit 1` after inspecting the first result.
Without `--explain_all`, the existing three-method workflow is used.

Other targets:

```bash
python src/scripts/explanations/OC_CelebA.py \
  --config CelebAHQ_epito --attribute Smiling \
  --show-gt --limit 1 --explain_all --no-heatmaps --no-auc

python src/scripts/explanations/OC_CelebA.py \
  --config CelebAHQ_epito --attribute Eyeglasses \
  --show-gt --limit 1 --explain_all --no-heatmaps --no-auc
```

If no target is specified, `Brown_Hair` is explained, matching the notebook's
index 11. `--target-class N` selects a numeric output instead. Do not combine it
with `--attribute`. The top predicted attribute is shown separately in the
suptitle, since it may differ from the selected attribute.

Outputs keep attributes separate:

```text
results/explanations/OC_CelebA/<partition-folder>/Brown_Hair/bg_gray_B1000/<image-id>/
results/evaluations/OC_CelebA/<partition-folder>/Brown_Hair/bg_gray_B1000/
```

## Model and ground-truth matching

Preprocessing uses a direct 224×224 bilinear resize and the notebook's mean
`[0.5063, 0.4258, 0.3832]` and standard deviation `[0.2644, 0.2436, 0.2397]`.
Explanations retain the source image/partition dimensions; gradient maps are
mapped back from the model's resized input. PIL resizing may have slightly
different rounding from the notebook's initial OpenCV resize.

The default output order is the standard 40 CelebA attributes. To read the
header from your dataset instead, add:

```bash
--attribute-file /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/data/data_full/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-attribute.txt
```

`--show-gt` selects relevant part masks:

| Attribute | Localization proxy |
| --- | --- |
| Brown_Hair, Black_Hair, Blond_Hair, Gray_Hair, Straight_Hair, Wavy_Hair | hair |
| Smiling | lower lip, mouth, upper lip |
| Eyeglasses | eye_g |
| Wearing_Hat | hat |
| Big_Nose, Pointy_Nose | nose |

These part masks are **attribute-region proxies**, not per-pixel attribute truth
(e.g. a hair mask does not establish that hair is brown). This scope is recorded
in GT metadata. The runner does not filter to attribute-positive images.
Attributes with no part mapping, such as `Male` or `Young`, receive an explicit
unmapped status instead of an invented GT mask. `--gt-class hair` or another
part name explicitly overrides automatic matching. Missing part files are
reported as unavailable. GT is drawn as black boundaries; there is no standalone
GT column in the all-method figure.

## Validation

The published checkpoint was downloaded and loaded strictly against the notebook
architecture. A CPU all-method smoke run completed **13/13 methods** with reduced
budgets for Brown_Hair and Smiling on a bundled astronaut image, synthetic
hair/mouth GT, and quadrant partitions.
Artifacts are under `results/celeba_xai_smoke/`; they are execution fixtures,
not CelebA localization-accuracy results or EPITO GPU validation.

Tests cover strict state loading, logits rather than softmax, preprocessing,
attribute header order, the default Brown_Hair target, GT proxies, LRP conversion,
separate attribute output paths, and checksum failures without overwriting weights.
