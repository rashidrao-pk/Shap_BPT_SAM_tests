# FunnyBirds

[FunnyBirds](https://github.com/visinf/funnybirds) ([paper](https://arxiv.org/abs/2308.06248)) is a synthetic dataset built to evaluate XAI methods. A bird's class is defined only by its parts, so part masks mark evidence the model must use to separate classes. This is stronger than a natural-image mask, which only says where the object is.

| | |
| --- | --- |
| Images | 50 classes, 50,000 train, 500 test (10 per class), 256×256 |
| Ground truth | Per-pixel part maps: beak, eye, foot, tail, wing, plus body |
| Model | [FunnyBirds ResNet-50](../MODELS/FUNNYBIRDS_RESNET50.md) |
| Download | 1.55 GB |

## 1. Download the dataset

On a node with internet access, from the repository root:

```bash
python src/scripts/download_datasets.py --dataset funnybirds --assets dataset
```

The dataset goes to `/beegfs/home/mrashid/datasets/FunnyBirds` (change the parent with `--root`). The archive holds about 127,000 files, so extraction is slow on a network filesystem.

## 2. Models

- [FunnyBirds ResNet-50](../MODELS/FUNNYBIRDS_RESNET50.md)

## 3. Next steps

1. [Generate partitions](../Partitions/DATASETS/FUNNYBIRDS.md)
2. [Run explanations](../XAI/DATASETS/FUNNYBIRDS.md)

## Limits on what you can claim

- Only the parts that differ from a competing class actually separate it, so the union of all parts is an upper bound on the true evidence, not an exact target.
- Part maps use exact colours. Anti-aliased edge pixels (about 3% of part pixels in a 12-image sample) stay unlabeled, as in the authors' code. Background distractors are excluded.
- This repository reports insertion/deletion AUC and GT localization. It does **not** implement the paper's FunnyBirds protocols (CSDC, PC, DC, D, SD, TS). Those use the `test_interventions` renders, so do not compare scores with the paper's table.
- Results are on synthetic images and do not transfer directly to natural images.
