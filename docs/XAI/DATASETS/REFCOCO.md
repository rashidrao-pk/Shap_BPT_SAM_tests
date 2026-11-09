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

The three partition methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` adds LIME, Shap-AA, IDG, GradExp, GradShap and GradCAM; LRP is reported as unsupported. Add `--dry-run` to check paths without loading CLIP.

| Option | Default | Meaning |
| --- | --- | --- |
| `--all-sentences` | off | Explain every expression of each object, not only the first |
| `--image-ids` | all | Record IDs (`<image>__ref<r>_s<s>`) or image IDs (all their objects) |
| `--limit` | all | Records; one object per image first, so `--limit 20` covers 20 photos |

## Known problem: raw cosine similarity has almost no signal

Measured locally on 5 expressions from 2 val images (CLIP ViT-B/32, 224×224):

| Expression | Photo | Black | Gray | White | Blurred | Noise |
| --- | --- | --- | --- | --- | --- | --- |
| person standing | 0.210 | 0.262 | 0.233 | 0.224 | 0.222 | 0.246 |
| person in bottom right corner | 0.222 | 0.253 | 0.220 | 0.214 | 0.232 | 0.249 |
| man on right | 0.227 | 0.224 | 0.204 | 0.206 | 0.223 | 0.206 |
| monitor far right | 0.234 | 0.248 | 0.227 | 0.218 | 0.244 | 0.248 |
| computer monitor | 0.256 | 0.254 | 0.231 | 0.218 | 0.249 | 0.253 |

A blank image often matches the phrase as well as, or better than, the photo. With the default `full` background, the fully masked score was 0.237 for both images, against 0.210 and 0.234 for the photos. The explained difference is then about zero or negative, the BPT and SAM_c maps came out at a scale of 1e-6, and AUC normalised by that difference is unstable. Do not run the full set with this score until it is changed. Options under consideration:

1. **Softmax over the image's expressions:** how much more the photo matches this expression than the expressions for the image's other referred objects. The shared offset cancels, and a blank image scores 1/k. RefCOCO images always hold at least two objects of the same category, so k ≥ 2.
2. **Cosine minus a neutral prompt:** similarity to the expression minus similarity to "a photo", which removes the blank-image bias.
3. **Gray background only,** as FIxLIP masks with the mean colour: the photo then beats the blank image for 4 of 5 expressions, but only by 0.002 to 0.025.

## What is explained

For each record, the text is the referring expression, fixed. The score is CLIP's cosine similarity between the image and that text, as in [FIxLIP](https://shapiq.readthedocs.io/en/latest/demos/fixlip/index.html). Masking image regions changes the similarity; the map shows which regions raise or lower it. FIxLIP also explains the text tokens and their interactions with patches; here only the image is explained.

In the outputs, `explained_class` is the expression, and `<record>_predictions.json` also holds the object's annotation ID, category and image.

## Ground truth

`--show-gt` shows the mask of the object the expression refers to. Results include pointing accuracy and best IoU. Best IoU picks the best threshold using GT, so it is a diagnostic, not a test score.

## Reading the results

- CLIP scores the whole photo against the phrase; it is not a grounding model. A map that highlights the other man in "man on right" shows what CLIP used, not a bug in the method.
- Compare expressions that name only appearance with those that rely on location ("left", "right"); CLIP is known to be weak on the second kind.
- The similarity range is narrow (about 0.1 to 0.35). AUC is normalised by the scores of the full and the fully masked image, so it remains comparable.
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
