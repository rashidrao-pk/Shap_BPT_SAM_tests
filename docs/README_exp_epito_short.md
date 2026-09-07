# Experiment Matrix

This document tracks the full-image-set XAI experiments for MS-COCO images using
the BPT/SAM partition variants.

Each full-run script saves:


## E1: YOLOv11 Detection

Use this for object-detection explanations on MS-COCO. The score vector is built
from detected class confidences. Use `--class-aggregation sum` to sum detections
per class or `--class-aggregation max` to keep only the maximum confidence per
class.

Smoke test:

```bash
python src/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --max-evals 500 \
  --eval-batch-size 2048 \
  --auc-batch-size 1024 \
  --verbose-level medium \
  --model yolo11s \
  --bg-type noise \
  --class-aggregation sum
```

Default result folder for `noise` and `max_evals=500`:

```bash
python src/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --max-evals 500 \
  --eval-batch-size 2048 \
  --auc-batch-size 1024 \
  --verbose-level medium \
  --model yolo11s \
  --bg-type noise \
  --class-aggregation sum
```

Use `--no-budget-suffix` to restore the older two-part YOLO folder name such as
`xai_results_E1_1`.

Nonstandard budgets use an explicit suffix, for example `--max-evals 750`
creates `xai_results_E1_1_custom750`.
---------------------------------------------------------------------------------------------------------
## E2: ResNet-50 Classification

Use this for ImageNet classification explanations on the same MS-COCO images.
The score vector is the ResNet-50 softmax output over 1000 ImageNet classes.

Smoke test:

```bash
python src/scripts/run_resnet_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --verbose-level medium \
  --bg-type noise \
  --eval-batch-size 512 \
  --auc-batch-size 256 \
  --resnet-batch-size 512
```


```bash
python src/scripts/run_resnet_full.py \
  --config MSCOCO_epito \
  --verbose-level medium \
  --bg-type noise \
  --eval-batch-size 2048 \
  --auc-batch-size 1024 \
  --resnet-batch-size 2048
```

## E3: ViT-B/16 Classification

Use this to compare a transformer classifier against ResNet-50. The score vector
is the ViT-B/16 softmax output over 1000 ImageNet classes.

Smoke test:

```bash
python src/scripts/run_vit_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --verbose-level medium \
  --bg-type noise \
  --eval-batch-size 512 \
  --auc-batch-size 256 \
  --vit-batch-size 512
```

Default result folder for `noise`:

```text
xai_results_E3_1
```

```bash
python src/scripts/run_vit_full.py \
  --config MSCOCO_epito \
  --verbose-level medium \
  --bg-type noise \
  --eval-batch-size 2048 \
  --auc-batch-size 1024 \
  --vit-batch-size 2048
```

## E4: DETR Detection

Use this to compare YOLOv11 with a transformer-based object detector. The score
vector is built from DETR post-processed detection scores per class. Like YOLO,
DETR supports:

- `--class-aggregation sum`
- `--class-aggregation max`

DETR uses Hugging Face `transformers`. Install it in the runtime environment if
needed:

```bash
pip install transformers
```

Smoke test:

```bash
python src/scripts/run_detr_full.py \
  --config MSCOCO_mac \
  --limit 2 \
  --max-evals 50 \
  --verbose-level medium \
  --bg-type noise \
  --threshold 0.3 \
  --detr-batch-size 4 \
  --eval-batch-size 16 \
  --auc-batch-size 8
```

Default result folder for `noise`:

```text
xai_results_E4_1
```

```bash
python src/scripts/run_detr_full.py \
  --config MSCOCO_epito \
  --max-evals 500 \
  --verbose-level medium \
  --bg-type noise \
  --threshold 0.3 \
  --detr-batch-size 64 \
  --eval-batch-size 64 \
  --auc-batch-size 64
```

## Summarizing Results

Generate boxplots, the full HTML report, and the outlier report from saved
results:

```bash
python src/scripts/summarize_results.py \
  --results-dir /path/to/xai_results_E1_1 \
  --boxplots \
  --html-report \
  --outlier-report
```

For intermediate runs, the summarizer reads:

```text
auc_results_all_images_current_run.csv
```

This file is updated after every completed image.

## No-Ground-Truth Evaluation Metrics

The aggregate AUC CSV also includes model-faithfulness metrics that do not use
segmentation ground truth:

| Metric                       | Meaning                                                                                                            |
| :--------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| `drop_at_10/20/30`           | Normalized confidence drop after removing the top attributed pixels. Higher is better.                             |
| `insert_at_10/20/30`         | Normalized confidence recovered by inserting only the top attributed pixels into the background. Higher is better. |
| `sufficiency_at_10/20`       | Model confidence when only the top attributed pixels are kept. Higher is better.                                   |
| `sufficiency_gap_at_10/20`   | `f_S - sufficiency_at_k`. Lower is better.                                                                         |
| `comprehensiveness_at_10/20` | `f_S - score_after_removing_top_k`. Higher is better.                                                              |
| `sensitivity_n_corr`         | Correlation between attribution mass removed and actual model confidence drop over random masks. Higher is better. |
| `stability_top10_jaccard`    | Top-10% attribution rank stability under small attribution perturbations. Higher is better.                        |

## Recommended Core Matrix

At minimum, run the following for the main comparison:

|  Exp No  | Model     | Task           | Background |
| :------: | :-------- | :------------- | :--------- |
| `E1_1_4` | YOLOv11   | Detection      | `noise`    |
|  `E2_1`  | ResNet-50 | Classification | `noise`    |
|  `E3_1`  | ViT-B/16  | Classification | `noise`    |
|  `E4_1`  | DETR      | Detection      | `noise`    |

Then add the background ablation for YOLOv11:

```text
E1_1_4, E1_2_4, E1_3_4, E1_4_4, E1_5_4
```

Then add the budget ablation for YOLOv11:

```text
E1_1_1, E1_1_2, E1_1_3, E1_1_4, E1_1_5
```

## RUN EXPs

```bash
/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/src/scripts/prepare_setup.sh
```


