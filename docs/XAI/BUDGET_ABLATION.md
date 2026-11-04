# Select one common budget with a small ablation

Normal experiments now use **one budget** for BPT, BPT+SAM_c and
SAM_c+KernelSHAP. A separate pilot tests candidate budgets on a few examples.
It can justify the **smallest tested budget that passes the declared checks**;
it cannot prove a universal minimum from a few images. Existing numeric defaults
are provisional, not empirical recommendations.

## 1. Run a pilot per dataset/model

For ImageNet/ResNet:

```bash
python src/scripts/explanations/budget_ablation.py \
  --runner OC_ResNet --config ImageNet_epito \
  --show-gt --gt-only --limit 5 --bg-type black \
  --budgets 100 200 500 1000 2000 --reference-budget 4000 \
  --repeats 3 --auc-num-samples 200
```

The same script supports:

| Runner    | Config         | Additional options                                              |
| --------- | -------------- | --------------------------------------------------------------- |
| OC_ResNet | ImageNet_epito | `--show-gt --gt-only`                                           |
| OC_ViT    | ImageNet_epito | `--show-gt --gt-only`                                           |
| COD_ViT   | CAMO_epito     | `--show-gt`                                                     |
| OD_DETR   | MSCOCO_epito   | `--show-gt`                                                     |
| OD_yolo   | MSCOCO_epito   | `--show-gt`; optional `--checkpoint src/checkpoints/yolo11s.pt` |
| OC_CelebA | CelebAHQ_epito | `--show-gt --attribute Brown_Hair`                              |

Keep budgets, repeats, criteria and AUC resolution the same in all pilots. Use
`--bg-type black` explicitly if you want that background throughout. Other
backgrounds or CelebA attributes need their own pilot scopes.

Up to five eligible images are sampled reproducibly using `--pilot-seed 0`, with
**three calibration and two held-out validation images**. The split is saved
before model inference. Explicit `--image-ids` define the eligible pool, still
split into calibration/validation. At least three images are required. Use
`--dry-run` first to inspect IDs and paths without loading weights.

The pilot uses saved SAM_c partitions and each runner's existing scoring,
backgrounds and fixed explained class. Only the three budget-dependent methods
are swept: LIME, AA and gradient baselines are not repeatedly recomputed.
AUC is computed at 200 requested points; repeated masks can give fewer points.
Model loading and BPT building are shared across the candidate budgets. Budgets
are nominal method-specific evaluation/sample limits; actual query counts and
background-expanded model-image counts are recorded rather than assumed equal.

## 2. Inspect the evidence

Outputs are separate from the main experiments:

```text
results/evaluations/budget_ablation/<runner>/<partition>/bg_black/pilot/
```

CelebA adds its attribute before `pilot/`. Open `budget_report.html`:

- `budget_heatmaps.png` per image compares repeat-mean maps at every budget,
  using a common reference-based color scale within each method row.
- `budget_curves.png` shows worst observed agreement/error and median runtime.
- `budget_comparisons.csv` contains every image/method/budget/repeat, actual mask
  evaluations and background-expanded model-image counts.
- `reference_diagnostics.json` checks repeat uncertainty of the high-budget maps.
- `selection.json` records the decision, scopes, criteria and held-out result.

Default **empirical tolerances**, configurable before running:

| Check versus the repeat-mean high-budget reference | Requirement |
| -------------------------------------------------- | ----------- |
| Signed-map Spearman rank agreement                 | ≥ 0.95      |
| Tie-aware top-10% absolute-attribution Jaccard     | ≥ 0.90      |
| Relative L1 attribution error                      | ≤ 0.10      |
| Insertion and deletion AUC differences             | each ≤ 0.01 |

Every repeat must pass. By default every image must pass, separately for each
method and pilot scope. A candidate must also pass at **all larger tested
budgets**, preventing selection of an isolated lucky result. Constant or unstable
references do not count as evidence of convergence. The reference uses independent
seeds; its budget is an approximation, not exact Shapley ground truth.

Choose the smallest passing budget on calibration images only. Validate that
candidate once on held-out images. A held-out failure leaves `selected_budget`
null; it does not choose a larger budget using validation outcomes. GT best-IoU
is an oracle-threshold diagnostic, not a selection criterion. A `--no-auc` pilot
is diagnostic only and cannot produce a production budget-selection file.

## 3. Select one budget across all pilots

Combine the **completed pilot directories** printed by each run:

```bash
python src/scripts/explanations/budget_ablation.py \
  --combine /path/to/resnet/pilot /path/to/vit/pilot /path/to/camo/pilot \
            /path/to/detr/pilot /path/to/yolo/pilot /path/to/celeba/pilot \
  --output-dir results/evaluations/budget_ablation/common
```

The common budget must pass independently in every pilot/method group; results
are not pooled into a mean that hides weaker datasets. Missing comparisons or
incompatible designs block selection. Include every intended dataset/model in
this command before describing the result as common across those scopes.

## 4. Use that budget for the main experiments

```bash
python src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only --limit 1 \
  --explain_all --bg-type black --no-heatmaps --no-resume \
  --budget-selection results/evaluations/budget_ablation/common/selection.json
```

Use the same selection file with the other runners. Application checks the
runner/background scope and the AUC resolution. Unvalidated, partial-method or
stability-only recommendations are refused. Keep the same model/checkpoint,
partition settings, attribute and scoring setup as the pilot. Report final
quality on independent images, separately from calibration and pilot validation.

For a manual single budget, use `--budget N` (alias `--max-evals N`). With
`--explain_all`, this produces only one map per partition-based method. LIME-1000,
Shap-AA-1000 and gradient settings remain fixed. Explicit `--partition-budgets`
manual sweeps remain available for compatibility; they are not the normal default.

Cache files allow an interrupted pilot to resume. Changing settings or input
stamps requires a new `--study-name`. Timings in resumed caches are the original
measurements; compare runtimes on the same hardware. The saved metadata records
versions, input stamps, model-file stamps and hardware. Do not tune tolerances
or candidate grids on held-out failures and then present the reused hold-out as
independent validation; use fresh held-out examples for a revised study.

The small pilot supports an empirical choice within its tested grid and scopes,
not a population-level guarantee. More diverse held-out examples are needed for
stronger generalization claims.

References for metric/method definitions:
[Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html),
[Kernel SHAP sampling](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html).
The thresholds above are project criteria, not thresholds prescribed by these sources.
