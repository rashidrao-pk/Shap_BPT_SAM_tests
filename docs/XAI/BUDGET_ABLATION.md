# Interactive XAI budget ablation: YOLO + SAM1

Test **BPT** and **BPT+SAM_c** only. KernelSHAP is excluded. Use **full** background replacement throughout. Run these Python commands yourself inside an allocated EPITO GPU session; no batch job is submitted by this workflow.

The previous three-image pilot finished. Its compatible BPT maps can be reused, so you do not need to repeat that inference. Results from this two-method workflow are saved in a new sibling folder named `interactive_bpt_r8000`; the earlier study remains intact.

## 1. Select the existing source study

After pushing the code changes and pulling them on EPITO, activate `pt_312_sam` and work from the repository root:

```bash
export SOURCE_STUDY="$PWD/results/evaluations/budget_full_20261105_v1/budget/OD_yolo/MSCOCO_sam1_vit_b_62_compact-sam_200/bg_full/full_budget"

python -u src/scripts/explanations/budget_interactive.py \
  --source-study "$SOURCE_STUDY" --images 3 --dry-run
```

This preserves the source's fixed 20 images, original 12-calibration/8-validation split, YOLO checkpoint, SAM1 partitions and background. The new candidate grid is **1000, 2000, 4000, 6000**, with **8000 as the reference**. Reuse checks source inputs, checkpoint stamps, package versions and scoring settings before loading a model. The new reference uses a separate study name. Cached maps at matching budgets can be reused; all map comparisons and reference means are recomputed against 8000. The old 4000 reference may be reused as a 4000 candidate for deterministic BPT, never as an 8000 reference. Changing image/model/scoring settings prevents reuse. If the scorer implementation itself changes, do not reuse old evaluations.

## 2. Run a small checkpoint

```bash
python -u src/scripts/explanations/budget_interactive.py \
  --source-study "$SOURCE_STUDY" --images 3
```

The terminal prints **REUSED**, **CACHED**, or **DONE** for every method/budget/repeat, followed by explanation time, evaluation time and AUC scores for newly computed work. It prints an image completion message and saves an intermediate conclusion after each calibration image. The original three repetitions are retained; repeated deterministic BPT maps are not independent statistical samples.

Open the printed `interim_conclusion.md` or `interim_summary.csv`. These show observed agreement, errors and timings, and may identify a descriptive candidate. They do **not** validate a production budget.

## 3. Increase calibration examples when you choose

```bash
python -u src/scripts/explanations/budget_interactive.py \
  --source-study "$SOURCE_STUDY" --images 6

python -u src/scripts/explanations/budget_interactive.py \
  --source-study "$SOURCE_STUDY" --images 12
```

Run one command at a time. Previously completed maps are reused. Keep the same study name and source. Ctrl-C is safe: rerun the identical command to reuse completed evaluations; only interrupted work is repeated.

These stages use only the original calibration images. Intermediate inspection is exploratory: do not interpret repeated checks as repeated independent significance tests, and do not change tolerances or select favorable images after seeing their outcomes.

## Optional: refine the budget range

Use the calibration results to decide which interval needs more points. For example, add 2500 and 3000 while retaining 8000 as reference:

```bash
python -u src/scripts/explanations/budget_interactive.py \
  --source-study "$SOURCE_STUDY" --images 3 \
  --budgets 1000 2000 2500 3000 4000 6000 \
  --reference-budget 8000 --study-name interactive_bpt_refined_r8000
```

This command works directly from the original source study; a coarse run is not required first. If you already ran the coarse stage and `${SOURCE_STUDY%/*}/interactive_bpt_r8000/study.json` exists, you may use that folder as the source to reuse its 8000-reference evaluations too. Merely exporting `COARSE_STUDY` does not create that study. Changing a grid requires a new study name; existing results are not overwritten. The three repetitions and image split stay fixed. You can similarly add 5000 or 7000 if that is the unresolved interval. Every candidate must be strictly below 8000.

Record the refinement rule before collecting new calibration results. Keep the held-out images untouched. After choosing a grid, repeat exactly the same source, grid, reference and study-name arguments for later stages and `--final`; do not silently switch back to the coarse defaults.

## 4. Final paired non-inferiority test

When you decide to finish the predeclared design:

```bash
python -u src/scripts/explanations/budget_interactive.py \
  --source-study "$SOURCE_STUDY" --final
```

This completes all 20 fixed images and runs the paired-bootstrap analysis, including the eight held-out images. Run the final confirmation once; a held-out failure is not permission to retune and reuse the same hold-out as independent validation.

Read the printed **`statistical_conclusion.md`**. Full bounds are in `statistical_bounds.csv` and the decision is in `statistical_selection.json`. The separate `conclusion.md`/`selection.json` are the older strict per-image diagnostic; the statistical conclusion is the relevant decision for this workflow.

## Test and interpretation

Following the attached ChatGPT discussion, require evidence that a smaller budget is sufficiently close to the high-budget reference, rather than treating a nonsignificant difference as equivalence.

Default coarse budgets: **1000, 2000, 4000, 6000**. Approximate reference: **8000**. AUC resolution stays fixed at 200; it is not swept.

| Paired loss against the reference        | Acceptable mean loss |
| ---------------------------------------- | -------------------: |
| Relative L1 attribution error            |                 0.10 |
| 1 − signed Spearman correlation          |                 0.05 |
| 1 − top-10% absolute-attribution Jaccard |                 0.10 |
| Absolute insertion AUC difference        |                 0.01 |
| Absolute deletion AUC difference         |                 0.01 |

For each loss, test **H0: mean loss ≥ margin** against **H1: mean loss < margin**. Average repeats within each image, resample paired images 20,000 times, and form simultaneous one-sided 95% upper bounds across budgets, both methods and metrics. A budget passes when every bound is below its margin, reference checks pass, and all larger tested budgets also pass. Choose on calibration, then accept or reject once on the held-out set without escalating the budget from validation outcomes.

These bootstrap bounds have approximate coverage and the pilot is small. A positive result means mean agreement with an approximate reference under the tested settings, not exact Shapley values, ground-truth correctness, or a guarantee for every image. No passing budget means **no budget is validated**; do not automatically choose 4000 or 6000.

Any recommendation applies only to **BPT and BPT+SAM_c on the tested YOLO/SAM1 MS-COCO setup**. It does not cover KernelSHAP, LIME, gradients, SAM2, DETR or other datasets. The ordinary all-method XAI runner still includes other methods; this ablation does not supply a validated budget for them.
