# Budget steps: how much does AUC still change?

This study explains each image at a list of budgets and measures how much AUC+ (insertion) and AUC- (deletion) change from one budget to the next. Once the change is small, a larger budget adds little.

For one image and one step:

```text
ΔAUC = | AUC(next budget) - AUC(previous budget) |
```

The plot shows the mean over images, with AUC+ and AUC- in separate panels. Absolute values are used so that a rise on one image and a fall on another do not cancel. Signed changes are also saved.

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
