# Budget steps: how much does AUC still change?

This study explains each image at a list of budgets and measures how much AUC+ (insertion) and AUC- (deletion) change from one budget to the next. Once the change is small, a larger budget adds little.

For one image and one step:

```text
ΔAUC = | AUC(next budget) - AUC(previous budget) |
```

The plot shows the mean over images, with AUC+ and AUC- in separate panels. Absolute values are used so that a rise on one image and a fall on another do not cancel. Signed changes are also saved.

> **YOLO results before 2026-11-17 are invalid.** YOLO received RGB arrays where Ultralytics expects BGR, so red and blue were swapped. This is fixed (`yolo_io.py`, input version 2). The YOLO signature now includes the input version, so the old cached maps are recomputed automatically when the study is repeated. The MS-COCO/YOLO numbers recorded earlier (settled budgets 5000 and 7000) must be redone:
>
> ```bash
> python src/scripts/explanations/budget_steps.py --budget-steps 20 50 100 500 1000 2000 3000 4000 5000 6000 7000 8000 --limit 20
> ```

It differs from [the budget ablation](BUDGET_ABLATION.md), which compares each budget with a high-budget reference map. This study needs no reference, so it is much cheaper, but small steps show only that AUC has stopped moving, not that the map is correct.

## Run it

Defaults are YOLO11s, MS-COCO, SAM-1 partitions, full background, 5 images, budgets 20, 50, 100, 500, 1000 and 2000, and methods BPT and BPT+SAM_c. From the repository root, on a GPU node:

```bash
python src/scripts/explanations/budget_steps.py --dry-run   # Shows the images and output folder
python src/scripts/explanations/budget_steps.py
```

Choose the images and budget steps:

```bash
python src/scripts/explanations/budget_steps.py --limit 5 --budget-steps 20 50 100 500 1000 2000 3000 4000 5000 6000 7000 8000
python src/scripts/explanations/budget_steps.py --limit 1 --budget-steps 50 100              # Quick check
python src/scripts/explanations/budget_steps.py --limit 20 --budget-steps 100 200 500 1000 2000 4000
python src/scripts/explanations/budget_steps.py --image-ids 000000044279 000000118209 --budget-steps 100 500 1000
```

| Option                 | Effect                                                                    |
| ---------------------- | ------------------------------------------------------------------------- |
| `--limit N`            | Number of images, any positive number; the default is 5                   |
| `--budget-steps A B C` | Budgets to test, giving steps A→B and B→C; `--budgets` is the same option |
| `--image-ids ...`      | Use exactly these images, in this order; `--limit` then keeps the first N |
| `--methods ...`        | Any of `BPT`, `BPT+SAM_c`, `SAM_c+KernelSHAP`                             |
| `--tolerance X`        | Change at or below which a step counts as settled; the default is 0.01    |
| `--study-name NAME`    | Output folder name; use a new name to keep an earlier plot                |

Finished maps are reused, so a larger `--limit` only computes the new images, and a new budget only computes that budget. A larger limit keeps the same images and adds more. Tables and plots show only the current command's images and budgets, and they replace the earlier ones unless you set `--study-name`. With one image there are no error bars.

For another dataset, pass its runner and config, for example `--runner OC_ResNet --config ImageNet_epito`. Repeat a command to resume it. Plots and tables are refreshed after each image.

## Choose images by difficulty

By default the images are a random sample. To base the budget on harder images, select them by difficulty:

```bash
# Equal numbers of easy, medium and hard images (recommended)
python src/scripts/explanations/budget_steps.py --budget-steps 100 500 1000 2000 3000 4000 5000 6000 8000 10000 \
  --limit 30 --select stratified

# Only the hardest third: a worst-case budget
python src/scripts/explanations/budget_steps.py --budget-steps 100 500 1000 2000 3000 4000 5000 6000 8000 10000 \
  --limit 30 --select hard
```

**How difficulty is measured.** No model is needed; it uses the GT and the SAM_c partition of each candidate image:

| Feature | Harder when |
| --- | --- |
| `regions` (SAM_c cells) | more |
| `objects`, `categories` | more |
| `max_same_class` (instances of the most repeated class) | more |
| `smallest_object`, `mean_object` (share of the image) | smaller |
| `partition_fit` (mean best IoU of the partition with each GT object) | lower |

Each feature is ranked across all candidates. The difficulty score is the mean of those percentile ranks, from 0 (easy) to 1 (hard), and the candidates are split into thirds. `stratified` takes hard, medium and easy images in turn; `hard` takes only the top third. With either, a larger `--limit` keeps the images already chosen. Features are computed once and cached in `image_difficulty.csv`. Scoring about 1000 COCO images takes a few minutes. `--difficulty-pool 300` scores only a random 300.

**Where results go.** Explanations are cached in the `steps/` folder and shared by every selection, so an image already explained is not explained again. Reports go to `steps/stratified/` or `steps/hard/`:

| File | Contents |
| --- | --- |
| `image_difficulty_scored.csv` | Every candidate: features, score, stratum, and whether it was selected |
| `budget_steps_by_stratum.csv` | Step summary within each stratum |
| `settled_budget.json` | Settled budget overall and `by_stratum` |

The run also prints the settled budget for each stratum, for example "5000 overall; 7000 for hard images".

**Check that the features matter.** Whether these features predict slow convergence is an assumption. Test it on images a study has already run, without running anything new:

```bash
python src/scripts/explanations/budget_steps.py --budget-steps 20 50 100 500 1000 2000 3000 4000 5000 6000 7000 8000 \
  --check-difficulty 3000
```

For each method, metric and feature, this gives the Spearman correlation between the feature and |AUC(3000) − AUC(largest budget)|, the distance an image still has to go. It is saved as `difficulty_check_B3000.csv`. A feature whose sign matches "harder when" and is clearly away from 0 is useful. With about 20 images only strong correlations (|ρ| > 0.45) are distinguishable from noise. Use the same `--select` and `--study-name` as the study you are checking.

**Choosing a budget from strata.** `stratified` gives a budget that represents the dataset and shows how much harder images need. `hard` gives a conservative worst-case budget, which every image in the full run then pays for. Report which one you used.

## Outputs

Saved in `<evaluations>/budget_steps/<runner>/<partitions>/bg_<background>/steps/`:

| File                       | Contents                                                      |
| -------------------------- | ------------------------------------------------------------- |
| `budget_steps.png`         | E[\|ΔAUC\|] per budget step, one panel each for AUC+ and AUC- |
| `budget_auc.png`           | Mean AUC+ and AUC- at each budget                             |
| `budget_auc.csv`           | AUC, time, and model calls for each image, method, and budget |
| `budget_steps.csv`         | Signed and absolute change for each image, method, and step   |
| `budget_steps_summary.csv` | Mean and SD across images for each method and step            |
| `settled_budget.json`      | The settled budget for each method                            |
| `budget_time.png`          | Mean explanation time per image at each budget (log scale)    |
| `budget_timing.csv`        | Per method and budget: time mean, SD and total, AUC scoring time, model calls, ms per call |
| `timing_summary.json`      | Totals per method, and this run: model load, wall time, maps computed and cached |

## Timing

At the end (and after each image) the run prints a timing table: mean explanation and AUC scoring time per image for each method and budget, the total per method, and what this run did.

- Times come from each saved map, so maps finished in earlier runs keep the time they took then. If those ran on another GPU, mixed times are not comparable; check `device` in `timing_summary.json`.
- `this_run` counts only maps computed now; cached maps take no time. `wall_sec` includes model loading, tree building and AUC scoring.
- Explanation time excludes AUC scoring, which is reported on its own.

## Reading the plot

- Error bars are the SD across images, cut off at zero in `budget_steps.png`.
- Budget steps are not equal in size. Steps are shown evenly spaced, so compare them as steps and not as a rate per added sample.
- The dotted line is the tolerance, 0.01 by default, matching the budget ablation.
- A ring marks the step that starts at the settled budget: the smallest budget after which every later step, in both AUC+ and AUC-, has a mean change within tolerance. There is no ring if the last step is still above tolerance.
- BPT and BPT+SAM_c give identical maps when repeated, so each runs once and the error bars show only differences between images. KernelSHAP is random, so its steps also include sampling noise from the single `--seed`.
- The settled budget is descriptive. A 5-image run is a first look, and does not validate a budget.
