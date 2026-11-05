# MS-COCO: select one budget with a small ablation

Use a few MS-COCO examples to select **one common budget** for BPT, BPT+SAM_c, and SAM_c+KernelSHAP. Test both detectors (YOLO11s and DETR) and both partition sources (SAM1 and SAM2), then combine the evidence.

The result is the **smallest tested budget passing the declared checks**, not proof of a universal minimum. Until the pilot finishes and validates a choice, budgets such as 500 or 1000 remain provisional.

## 1. Prepare SAM1 and SAM2 configs

Both folders must contain completed metadata and `*_compact.npy` maps. Make separate configs so both variants can be run without editing a config while another experiment uses it:

```bash
cp src/configs/MSCOCO_epito.yaml src/configs/MSCOCO_epito_sam1.yaml
cp src/configs/MSCOCO_epito.yaml src/configs/MSCOCO_sam2_epito.yaml
```

Edit only `data.mask_dir_final` in the copies to match your completed folders:

| Config              | `data.mask_dir_final` from the existing EPITO runs |
| ------------------- | -------------------------------------------------- |
| `MSCOCO_epito_sam1` | `MSCOCO_sam1_vit_b_62_compact-sam_200`             |
| `MSCOCO_sam2_epito` | `MSCOCO_sam2_large_62_compact-sam_200`             |

Keep the COCO image/annotation paths and output roots the same. If your folder names differ, use those names instead. The ablation reads saved compact maps; it does not generate partitions.

Run inside your allocated EPITO GPU session with `pt_312_sam` activated. Have the YOLO11s and `facebook/detr-resnet-50` checkpoints available; existing runner checkpoint/cache lookup is reused.

## 2. Check the pilot selection first

```bash
python -u src/scripts/explanations/budget_ablation.py \
  --runner OD_yolo --model yolo11s --config MSCOCO_epito_sam1 \
  --show-gt --limit 5 --bg-type black --class-aggregation sum \
  --budgets 100 200 500 1000 2000 --reference-budget 4000 \
  --repeats 3 --pilot-seed 0 --holdout-fraction 0.4 \
  --auc-num-samples 200 --study-name mscoco_pilot --dry-run
```

This validates paths and prints the output folder and image split without loading model weights. Here `--limit 5` samples five eligible completed-partition images reproducibly, rather than taking the first five filenames. They are split into **three calibration and two held-out validation images** before inference.

Use the same image pool for all four pilots. If the two partition folders contain identical completed IDs, the seed selects the same examples. Otherwise, provide the same common eligible IDs with `--image-ids ID1 ID2 ID3 ID4 ID5` in every command. Inspect the dry-run splits before proceeding. At least three eligible images are required; `--limit 1` is not a valid ablation.

## 3. Run the four pilots

Keep budgets, repeats, seed, background, scoring, and AUC resolution identical. Only the three budget-dependent methods are swept; LIME and gradient baselines are not included in this pilot.

### YOLO11s with SAM1

```bash
python -u src/scripts/explanations/budget_ablation.py \
  --runner OD_yolo --model yolo11s --config MSCOCO_epito_sam1 \
  --show-gt --limit 5 --bg-type black --class-aggregation sum \
  --budgets 100 200 500 1000 2000 --reference-budget 4000 \
  --repeats 3 --pilot-seed 0 --holdout-fraction 0.4 \
  --auc-num-samples 200 --study-name mscoco_pilot
```

### YOLO11s with SAM2

```bash
python -u src/scripts/explanations/budget_ablation.py \
  --runner OD_yolo --model yolo11s --config MSCOCO_sam2_epito \
  --show-gt --limit 5 --bg-type black --class-aggregation sum \
  --budgets 100 200 500 1000 2000 --reference-budget 4000 \
  --repeats 3 --pilot-seed 0 --holdout-fraction 0.4 \
  --auc-num-samples 200 --study-name mscoco_pilot
```

### DETR with SAM1

```bash
python -u src/scripts/explanations/budget_ablation.py \
  --runner OD_DETR --model facebook/detr-resnet-50 --config MSCOCO_epito_sam1 \
  --show-gt --limit 5 --bg-type black --class-aggregation sum \
  --budgets 100 200 500 1000 2000 --reference-budget 4000 \
  --repeats 3 --pilot-seed 0 --holdout-fraction 0.4 \
  --auc-num-samples 200 --study-name mscoco_pilot
```

### DETR with SAM2

```bash
python -u src/scripts/explanations/budget_ablation.py \
  --runner OD_DETR --model facebook/detr-resnet-50 --config MSCOCO_sam2_epito \
  --show-gt --limit 5 --bg-type black --class-aggregation sum \
  --budgets 100 200 500 1000 2000 --reference-budget 4000 \
  --repeats 3 --pilot-seed 0 --holdout-fraction 0.4 \
  --auc-num-samples 200 --study-name mscoco_pilot
```

For a specific local YOLO file, add `--checkpoint src/checkpoints/yolo11s.pt` if that is its actual location. For local DETR, use `--checkpoint /path/to/huggingface/model_directory`. Keep the same checkpoint for the corresponding final experiments.

The target is each detector's top-scoring class on the unmasked input and stays fixed across all budgets/repeats for that image. Detectors can choose different classes on the same image; the study tests convergence for each detector's own target. COCO GT contours and optional IoU diagnostics refer to that target class. GT is not used to construct partitions or choose the budget.

Black background is explicit here. `--bg-type full` averages multiple background replacements and is more expensive; it requires a separate pilot scope. Budgets are method-specific evaluation/sample limits, so actual mask evaluations and background-expanded model-image counts are recorded rather than assumed equal.

## 4. Resume after interruption

Repeat the identical pilot command. Completed **image/method/budget/repeat maps and their evaluations** are cached. An interrupted map is recomputed; completed maps are reused. This is finer-grained than the normal XAI runner's completed-image resume.

For a six-hour batch allocation, prefix the same arguments with:

```bash
sbatch src/hpc/run_xai.sbatch src/scripts/explanations/budget_ablation.py \
  --runner OD_yolo --model yolo11s --config MSCOCO_epito_sam1 \
  --show-gt --limit 5 --bg-type black --class-aggregation sum \
  --budgets 100 200 500 1000 2000 --reference-budget 4000 \
  --repeats 3 --pilot-seed 0 --holdout-fraction 0.4 \
  --auc-num-samples 200 --study-name mscoco_pilot
```

Create `logs/` before submission. Do not use `--no-resume`. Changing study settings, selected inputs, checkpoint stamps, or the eligible image pool requires a new `--study-name`. Freeze the pilot inputs while collecting the evidence. Cached timings are the original measurements; compare timings on the same hardware.

## 5. Inspect evidence and combine

Each pilot prints its directory under:

```text
<output.evaluations>/budget_ablation/<runner>/<partition>/bg_black/mscoco_pilot/
```

Open `budget_report.html`. The directory also contains:

- `budget_heatmaps.png` in each image folder: repeat-mean maps at each budget with one reference-based color scale per method row.
- `budget_curves.png`: worst observed agreement/error and median runtime.
- `budget_comparisons.csv`: image/method/budget/repeat measurements and actual query counts.
- `reference_diagnostics.json`: uncertainty of the high-budget reference maps.
- `selection.json`: decision, criteria, scopes, and held-out result.

Default empirical tolerances, declared before running:

| Check against the repeat-mean high-budget reference | Requirement |
| --------------------------------------------------- | ----------- |
| Signed-map Spearman rank agreement                  | ≥ 0.95      |
| Tie-aware top-10% absolute-attribution Jaccard      | ≥ 0.90      |
| Relative L1 attribution error                       | ≤ 0.10      |
| Insertion and deletion AUC difference               | each ≤ 0.01 |

Every repeat and, by default, every image must pass separately for each method/scope. A candidate must also pass at all larger tested budgets, avoiding a choice based on an isolated lucky result. Constant or unstable references are rejected. Budget 4000 is an approximate reference, not exact Shapley ground truth. AUC requests 200 points; repeated masks can reduce the number of evaluated points.

The smallest passing calibration budget is tested on held-out images. A held-out failure leaves `selected_budget` null; it does not choose a larger budget using validation outcomes. A `--no-auc` pilot is diagnostic only and cannot produce a production budget selection. GT-selected best IoU is diagnostic, not a selection criterion.

Once **all four pilots finish**, combine their printed directories. For the default output root and folder names above:

```bash
python src/scripts/explanations/budget_ablation.py \
  --combine \
  results/evaluations/budget_ablation/OD_yolo/MSCOCO_sam1_vit_b_62_compact-sam_200/bg_black/mscoco_pilot \
  results/evaluations/budget_ablation/OD_yolo/MSCOCO_sam2_large_62_compact-sam_200/bg_black/mscoco_pilot \
  results/evaluations/budget_ablation/OD_DETR/MSCOCO_sam1_vit_b_62_compact-sam_200/bg_black/mscoco_pilot \
  results/evaluations/budget_ablation/OD_DETR/MSCOCO_sam2_large_62_compact-sam_200/bg_black/mscoco_pilot \
  --output-dir results/evaluations/budget_ablation/mscoco_common
```

Use your printed paths if they differ. The common budget must pass independently in all four pilots and all three methods. Missing/incompatible studies block selection. Inspect the combined `selection.json`: only `status: validated` with a non-null `selected_budget` is ready for production. Otherwise report that no tested common budget was validated; do not assume 1000 is adequate.

## Automatic conclusion after running

Each finished pilot and the combined report now print a conclusion and save **`conclusion.md`** alongside `selection.json`. The HTML report displays the conclusion at the top.

For the combined MS-COCO study, read:

```bash
cat results/evaluations/budget_ablation/mscoco_common/conclusion.md
```

A successful result states: **“Recommended budget: N for BPT, BPT+SAM_c, and SAM_c+KernelSHAP”**, using the budget actually validated by your runs. It lists the tested datasets, detectors, partition folders, backgrounds, and calibration/validation counts.

If validation fails, references are unstable, no candidate passes, or AUC was disabled, the conclusion explicitly says that no common production budget is recommended and explains the failure. A partial-method pilot cannot recommend a budget for all three methods. No numeric recommendation is invented before collecting the evidence. Use the combined conclusion to decide the common MS-COCO budget; a single pilot's conclusion applies only to its tested setup.

## 6. Apply the validated budget

For YOLO, for example:

```bash
python -u src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito_sam1 --model yolo11s --show-gt \
  --explain_all --bg-type black --class-aggregation sum --no-heatmaps \
  --auc-num-samples 200 \
  --budget-selection results/evaluations/budget_ablation/mscoco_common/selection.json
```

For DETR:

```bash
python -u src/scripts/explanations/OD_DETR.py \
  --config MSCOCO_epito_sam1 --model facebook/detr-resnet-50 --show-gt \
  --explain_all --bg-type black --class-aggregation sum --no-heatmaps \
  --auc-num-samples 200 \
  --budget-selection results/evaluations/budget_ablation/mscoco_common/selection.json
```

Repeat with `MSCOCO_sam2_epito` for SAM2. Do not add a conflicting `--budget`, `--partition-budgets`, or `--kernel-shap-samples` override. The selection supplies one budget to all three partition methods; LIME-1000, Shap-AA-1000, and gradient settings stay fixed. Keep the pilot model/checkpoint, partition setup, background, class scoring, and AUC resolution unchanged. Existing incompatible result settings may require a separate output directory.

This selection covers the tested **MS-COCO detectors and partition variants**. To claim a common budget across other datasets, also collect their pilots and include them when combining. Report final quality on independent images, excluding pilot calibration and validation images. Do not retune criteria on held-out failures and then call the reused hold-out independent validation.

References for metric/method definitions: [Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), [Kernel SHAP sampling](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html). The tolerances above are project criteria, not thresholds prescribed by these sources.

To measure whether 200 AUC points improve evaluation enough over 100, use the separate [saved-map AUC point comparison](AUC_POINTS_ABLATION.md). Keep its resolution fixed while collecting a budget-selection study.
