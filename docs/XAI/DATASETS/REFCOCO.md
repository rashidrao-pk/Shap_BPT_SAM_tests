# RefCOCO explanations (vision–language)

Run from the repository root on a GPU node. First [download the dataset](../../DATASETS/REFCOCO.md), get [CLIP ViT-B/32](../../MODELS/CLIP.md), and [generate partitions](../../Partitions/DATASETS/REFCOCO.md).

## Run

```bash
# Smoke test: one expression from each of 2 images
python src/scripts/explanations/VL_RefCOCO.py \
  --config RefCOCO_epito --show-gt --limit 2 --budget 1000 --no-heatmaps

# All methods (needs requirements-xai-all.txt)
python src/scripts/explanations/VL_RefCOCO.py \
  --config RefCOCO_epito --show-gt --limit 20 --budget 1000 --no-heatmaps --explain_all
```

### SAM version and compact partitions

The same command for each SAM model (`--sam`, its folder from the generator) and each saved compact file (`--compact-strategy`). Without `--compact-strategy`, the config's default `split` is used. Results of `split` go to `<partition-folder>_split`; each run prints `Partitions: ... | compact file: ...` first.

```bash
# SAM-1 ViT-B, greedy compact partitions
python src/scripts/explanations/VL_RefCOCO.py \
  --config RefCOCO_epito --sam sam1 --compact-strategy greedy --show-gt --limit 2 --budget 1000 --no-heatmaps

# SAM-1 ViT-B, split compact partitions
python src/scripts/explanations/VL_RefCOCO.py \
  --config RefCOCO_epito --sam sam1 --compact-strategy split --show-gt --limit 2 --budget 1000 --no-heatmaps

# SAM-2 Large, greedy compact partitions
python src/scripts/explanations/VL_RefCOCO.py \
  --config RefCOCO_epito --sam sam2 --compact-strategy greedy --show-gt --limit 2 --budget 1000 --no-heatmaps

# SAM-2 Large, split compact partitions
python src/scripts/explanations/VL_RefCOCO.py \
  --config RefCOCO_epito --sam sam2 --compact-strategy split --show-gt --limit 2 --budget 1000 --no-heatmaps
```

The three partition methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` adds LIME, Shap-AA, IDG, GradExp, GradShap and GradCAM; LRP is reported as unsupported. Add `--dry-run` to check paths without loading CLIP.

| Option | Default | Meaning |
| --- | --- | --- |
| `--all-sentences` | off | Explain every expression of each object, not only the first |
| `--image-ids` | all | Record IDs (`<image>__ref<r>_s<s>`) or image IDs (all their objects) |
| `--limit` | all | Records; one object per image first, so `--limit 20` covers 20 photos |

## Score (`--score`)

| Option | Output | Default |
| --- | --- | --- |
| `softmax-expressions` | Softmax, at CLIP's logit scale (100), over the cosine similarities between the image and every distinct referring expression of that image (all objects, all sentences); the probability of the record's own expression | Yes |
| `cosine` | Raw image–text cosine similarity, as in FIxLIP | |

Each record carries `image_expressions`, the softmax candidates. An image whose expressions are all identical cannot use the softmax; that record fails with a message, and `--score cosine` works for it. RefCOCO images always have at least two referred objects. The score is saved in `run_info.json` (`score_type`) and in each `<record>_predictions.json`.

### Measured behaviour (13 records from 5 val images, CLIP ViT-B/32, `full` background; 2026-11-17)

| Score | Photo beats chance (1/n) | Photo beats blank (f_S > f_0) | Blank vs 1/n, mean / max deviation |
| --- | --- | --- | --- |
| `cosine` | – | 7/13 | – (blank 0.19–0.24, photo 0.20–0.28) |
| `softmax-expressions` | 7/13 | 5/13 | 0.126 / 0.378 |
| Softmax calibrated by each expression's blank-image similarity (not implemented) | 5/13 | 5/13 | 0.026 / 0.075 |

- **Cosine** barely moves: the photo and the blank image differ by 0.00–0.07, often in the wrong direction.
- **The softmax** widens the range (for example 0.79 vs 0.06 for "the guy in the background…"). But a blank image does **not** score 1/n. At logit scale 100, small differences in how each phrase matches a blank image become large: generic phrases win ("man": 0.578 on a blank image, chance 0.2).
- **Calibration** (subtracting each expression's mean similarity over the five blank backgrounds before the softmax, as in contextual calibration) makes the blank score ≈ 1/n. It does not change how often the photo prefers the right expression.
- **The remaining failures are CLIP's.** On 6–8 of 13 records CLIP prefers another of the image's expressions even on the full photo, often for location phrases ("person left", "monitor far right"). No score built on CLIP ViT-B/32 fixes that. Explanations of those records show what drives CLIP away from the phrase. Report how many records have f_S > f_0.

## What is explained

For each record, the text is the referring expression, fixed. By default the score is the softmax probability of that expression among the image's expressions (see above); `--score cosine` uses the raw cosine similarity, as in [FIxLIP](https://shapiq.readthedocs.io/en/latest/demos/fixlip/index.html). Masking image regions changes the similarity; the map shows which regions raise or lower it. FIxLIP also explains the text tokens and their interactions with patches; here only the image is explained.

In the outputs, `explained_class` is the expression, and `<record>_predictions.json` also holds the object's annotation ID, category and image.

## Ground truth

`--show-gt` shows the mask of the object the expression refers to. Results include pointing accuracy and best IoU. Best IoU picks the best threshold using GT, so it is a diagnostic, not a test score.

## Reading the results

- CLIP scores the whole photo against the phrase; it is not a grounding model. A map that highlights the other man in "man on right" shows what CLIP used, not a bug in the method.
- Compare expressions that name only appearance with those that rely on location ("left", "right"); CLIP is known to be weak on the second kind.
- AUC is normalised by the scores of the full and the fully masked image. When f_S < f_0 the curve is flipped; when they are close, AUC is unstable. Check `f_S` and `f_0` in the predictions.
- Several expressions come from one image; report the number of images next to the number of expressions.
- CLIP's training data is not public; the photos may have been seen.

## ROAD and leaderboard

ROAD is scored during the run. For runs made without it:

```bash
python src/scripts/explanations/evaluate_road.py \
  --runner VL_RefCOCO --config RefCOCO_epito --budget 1000 --bg-type full --limit 20
```

The run ends with its leaderboard (`winner.md`). Across datasets, RefCOCO is the **vision-language** domain.

## Outputs and resume

```text
<output.explanations>/VL_RefCOCO/<partition-folder>/bg_full_B1000/<record-id>/
<output.evaluations>/VL_RefCOCO/<partition-folder>/bg_full_B1000/
```

Repeat the same command to resume: completed records are skipped.

## Not supported

The budget studies (`budget_ablation.py`, `budget_steps.py`) explain one score per image and do not yet set the expression per record, so RefCOCO is not registered there.

## Results so far

None yet. Only numbers produced by real runs belong here.
