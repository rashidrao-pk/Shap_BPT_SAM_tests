# FunnyBirds

[FunnyBirds](https://github.com/visinf/funnybirds) ([paper](https://arxiv.org/abs/2308.06248)) is a synthetic dataset built to evaluate XAI methods. A bird's class is defined only by its parts, so part masks mark evidence the model must use to separate classes. This is stronger than a natural-image mask, which only says where the object is.

| | |
| --- | --- |
| Images | 50 classes, 50,000 train, 500 test (10 per class), 256×256 |
| Ground truth | Per-pixel part maps: beak, eye, foot, tail, wing, plus body |
| Classifier | Authors' ResNet-50, trained on FunnyBirds train |
| Download | 1.55 GB dataset, 189 MB checkpoint |

## 1. Download the dataset and checkpoint

On a node with internet access, from the repository root:

```bash
python src/scripts/download_datasets.py --dataset funnybirds
```

Use `--assets dataset` or `--assets model` to fetch one of them. The dataset goes to `/beegfs/home/mrashid/datasets/FunnyBirds` (change the parent with `--root`) and the checkpoint to `checkpoints/funnybirds/`. The checkpoint is verified by SHA-256. The archive holds about 127,000 files, so extraction is slow on a network filesystem.

## 2. Why this checkpoint

Use the [authors' checkpoint](https://github.com/visinf/funnybirds-framework) rather than an ImageNet model. An ImageNet classifier has no FunnyBirds classes, so a part mask would say nothing about its decision.

The model takes 256×256 input scaled to 0–1 **with no ImageNet mean/std**. `OC_FunnyBirds` handles this. Do not run these images through `OC_ResNet`: with ImageNet preprocessing the model got 7 of 40 sampled test images right, against 80 of 80 with the correct preprocessing.

The authors also publish VGG16 and ViT-B/16 checkpoints. They are not wired in: only ResNet-50 has been checked against this code.

## 3. Generate partitions

```bash
python src/scripts/generate_funnybirds_partitions.py \
  --config FunnyBirds_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda --max-regions 62 --generate_at_least_n_masks 50 \
  --save-partitions sam compact
```

Add `--dry-run --limit 2` first to check paths. `data.image_dir` selects the split; keep `ground_truth.split` the same. Part masks are never used to build partitions.

## 4. Explain

```bash
python src/scripts/explanations/OC_FunnyBirds.py \
  --config FunnyBirds_epito --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --explain_all
```

`--show-gt` outlines the five class-defining parts. To select one, add `--gt-class beak` (also `eye`, `foot`, `tail`, `wing`). `--gt-class body` is available, but the body is shared by every class, so it is not class evidence.

## Limits on what you can claim

- Only the parts that differ from a competing class actually separate it, so the union of all parts is an upper bound on the true evidence, not an exact target.
- Part maps use exact colours. Anti-aliased edge pixels (about 3% of part pixels in a 12-image sample) stay unlabeled, as in the authors' code. Background distractors are excluded.
- This repository reports insertion/deletion AUC and GT localization. It does **not** implement the paper's FunnyBirds protocols (CSDC, PC, DC, D, SD, TS). Those use the `test_interventions` renders, so do not compare scores with the paper's table.
- Results are on synthetic images and do not transfer directly to natural images.
