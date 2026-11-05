# XAI budget ablation with full background replacement

Vary only the explanation budget: **100, 200, 500, 1000, 2000**, against a **4000** reference. Use `--bg-type full` for every run. This averages predictions over black, gray, white, blurred and noise replacements. AUC resolution stays fixed at 200; it is not swept.

The frozen study covers BPT, BPT+SAM_c and SAM_c+KernelSHAP with SAM1 and SAM2 partitions, across available ImageNet classifiers, COCO detectors, CAMO, CelebA Brown_Hair, BUSI SigLIP and PartImageNet ResNet. Each dataset contributes 20 shared images: 12 calibration and 8 held out. Targets are fixed per image/model. This scope does not cover other CelebA attributes or arbitrary future datasets.

## Statistical decision

The attached chat recommends paired non-inferiority: test whether a cheaper budget loses less than a predefined meaningful amount, rather than accepting it because a difference is not significant.

Here quality means agreement with the high-budget attribution reference. The predefined mean-loss margins are:

| Loss against the reference | Maximum acceptable mean loss |
| --- | ---: |
| Relative L1 map error | 0.10 |
| 1 − signed Spearman correlation | 0.05 |
| 1 − top-10% absolute-attribution Jaccard | 0.10 |
| Absolute insertion AUC difference | 0.01 |
| Absolute deletion AUC difference | 0.01 |

Average repeats within an image. Resample images, keeping budgets, models, methods and partition variants paired within each dataset. Use 20,000 bootstrap resamples and simultaneous one-sided 95% upper bounds across all tested budgets, scopes and metrics. The implementation uses the maximum centered bootstrap error after dividing each loss by its margin. This accounts for searching several budgets and comparing correlated methods. Bootstrap coverage is approximate, especially with a small hold-out.

Choose the smallest calibration budget whose upper bounds are below every margin, at that budget and all larger tested budgets. Confirm that choice once on held-out images. A held-out failure rejects it; it does not select a larger budget. Unstable or uninformative references prevent a recommendation. The existing strict per-image empirical analysis is retained as a separate diagnostic, not substituted for this statistical decision.

This establishes empirical mean convergence toward an approximation, not attribution correctness or a universal minimum. ShapBPT repeats check deterministic reproducibility; KernelSHAP references use independent seeds.

## Run on EPITO

From the repository, freeze the inputs and protocol before inference:

```bash
python3 src/hpc/prepare_full_budget_study.py \
  --repo "$PWD" \
  --study "$PWD/results/evaluations/budget_full_20261105_v1"
```

The submitted study uses frozen copies of the source, driver and batch file. Each six-hour allocation runs two GPU workers. Before timeout it queues a dependent continuation, up to 40 allocations. Completed map evaluations are reused; an interrupted evaluation is recomputed. Ordinary failures stop the driver and must be inspected. To stop the study, cancel its running job and any dependent continuation with `scancel`.

After all pilots finish, the driver automatically runs:

```bash
python src/scripts/explanations/budget_noninferiority.py \
  results/evaluations/budget_full_20261105_v1
```

Read `report/conclusion.md` and `report/noninferiority.json`. Until all scopes finish, no common budget is recommended. If none passes, report that result; do not assume 1000, 2000 or the 4000 reference is sufficient.

Method references: [SciPy paired bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) and [multiple-testing corrections](https://www.statsmodels.org/stable/_modules/statsmodels/stats/multitest.html). Project margins are declared here, not prescribed by these references.
