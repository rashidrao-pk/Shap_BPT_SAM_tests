# XAI budget ablation with full background replacement

**Current execution scope: MS-COCO with YOLO and SAM1 only, first three fixed images.** SAM2, DETR and other datasets are deferred. This stage reuses the two completed images and any cached work on the third. It stops after three images, with a six-hour cap and no automatic continuation.

Read the SAM1 pilot's `interim_conclusion.md`, `interim_summary.csv` and `interim_selection.json` under `budget/OD_yolo/MSCOCO_sam1_vit_b_62_compact-sam_200/bg_full/full_budget/`. This is a descriptive checkpoint, **not a validated production budget**. These images retain their original calibration/validation assignments and are not relabeled as independent hold-out evidence.

`manifest_yolo_sam1_quick.json` selects this short stage. The original 20-image design and all cached evaluations remain available. The execution option `--stop-after-images 3` limits work without changing cache identity; increase it later, or remove it to finish the original design. Images follow the original fixed order rather than a new sample selected from observed results.

Vary only the explanation budget: **100, 200, 500, 1000, 2000**, against a **4000** reference. Use `--bg-type full` for every run. This averages predictions over black, gray, white, blurred and noise replacements. AUC resolution stays fixed at 200; it is not swept.

The original frozen design covers BPT, BPT+SAM_c and SAM_c+KernelSHAP with SAM1 and SAM2 partitions, across available ImageNet classifiers, COCO detectors, CAMO, CelebA Brown_Hair, BUSI SigLIP and PartImageNet ResNet. Only the short SAM1 YOLO stage is currently scheduled. The original design has 20 shared images per dataset: 12 calibration and 8 held out. Targets are fixed per image/model. This scope does not cover other CelebA attributes or arbitrary future datasets.

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

The existing study has already been prepared. For the current three-image SAM1 stage, after its running job has stopped, resume with:

```bash
sbatch --job-name=YOLO-SAM1-pilot --gres=gpu:a100:1 --time=06:00:00 \
  --export=ALL,STUDY_MANIFEST=manifest_yolo_sam1_quick.json,STUDY_MAX_ALLOCATIONS=1 \
  results/evaluations/budget_full_20261105_v1/study.sbatch \
  "$PWD/results/evaluations/budget_full_20261105_v1"
```

The `active_manifest.txt` pointer selects the short SAM1 stage. The explicit `STUDY_MAX_ALLOCATIONS=1` disables automatic continuation. Do not launch either larger manifest while this run is active.

### See progress inside a pilot

The study controller prints a heartbeat every minute, including cached images, saved evaluation counts, current image, last logged method/budget/seed, time since the last save, and an approximate remaining evaluation time. The current job's main log also has a read-only monitor attached; inference is not restarted.

For a standalone snapshot on EPITO, without loading models:

```bash
python3 results/evaluations/budget_full_20261105_v1/budget_progress.py \
  results/evaluations/budget_full_20261105_v1
```

Add `--watch 60` for a refresh every minute. Each 20-image pilot has **1080 cached evaluations**: 20 images × 3 methods × 6 budget settings (including the reference) × 3 repeats. An image is cached only after all 54 evaluations are saved. The two pilots run in parallel, so their remaining times are not added together. ETA uses median timings for each method/budget and excludes tree building, repeat-mean reference AUC and reporting; it is approximate. A cached image is not itself a completed statistical analysis.

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
