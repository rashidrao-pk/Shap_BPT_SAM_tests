# Datasets

How to download each dataset. Checkpoints have their own pages in [Models](../MODELS/README.md).

| Dataset                         | Ground truth for XAI   | Model      | Were the explained images in training? | Runner            |
| ------------------------------- | ---------------------- | ---------------------- | -------------------------------------- | ----------------- |
| [FunnyBirds](FUNNY_BIRD.md)     | Class-defining parts   | [FunnyBirds ResNet-50](../MODELS/FUNNYBIRDS_RESNET50.md)     | No: test split                         | `OC_FunnyBirds`   |
| [PACO](PACO.md)                 | Object parts in scenes | [ResNet-50](../MODELS/RESNET50.md)     | Not via COCO; ImageNet not audited     | `OC_PACO`         |
| [Kvasir-SEG](KVASIR_SEG.md)     | Polyp masks            | [Kvasir v2 ViT](../MODELS/KVASIR_VIT.md)     | **Likely yes**                         | `OC_KvasirSEG`    |
| [ISIC 2018](ISIC_2018.md)       | Lesion masks           | [ISIC EfficientNet-B0](../MODELS/ISIC_EFFICIENTNET.md) | No shared image IDs                    | `OC_ISIC2018`     |
| [PartImageNet](PARTIMAGENET.md) | Object parts           | [ResNet-50](../MODELS/RESNET50.md)     | Possibly                               | `OC_PartImageNet` |
| [BUSI](BUSI.md)                 | Lesion masks           | [BUSI SigLIP](../MODELS/BUSI_SIGLIP.md)            | **Yes**, see its model card            | `OC_BUSI`         |
| [MS-COCO](MS_COCO.md)            | Instance masks         | [YOLO11s](../MODELS/YOLO.md), [DETR](../MODELS/DETR.md)          | No: val2017                            | `OD_yolo`, `OD_DETR` |
| [ImageNet](ImageNet.md)         | ImageNet-S class masks | [ResNet-50](../MODELS/RESNET50.md), [VGG16](../MODELS/VGG16.md), [ViT-B/16](../MODELS/VIT_B16.md)  | No: validation split                   | `OC_ResNet`, `OC_VGG`, `OC_ViT` |
| [CAMO](CAMO.md)                 | Camouflaged objects    | [ViT-B/16](../MODELS/VIT_B16.md)      | Not via CAMO; ImageNet not audited     | `COD_ViT`         |
| [CelebA-HQ](CelebAHQ.md)        | Face parts             | [CelebA CNN](../MODELS/CELEBA_CNN.md) | Unknown: see the notebook | `OC_CelebA` |

Models trained on the explained images are still valid to explain, since an attribution describes the model. They do not support claims about held-out performance, so state the overlap when reporting results.

## Quick start

From the repository root, on a node with internet access:

```bash
python src/scripts/download_datasets.py --dataset funnybirds --assets dataset   # or paco, kvasirseg, isic2018, all
```

Then get the [model checkpoints](../MODELS/README.md).

Then generate partitions and run the explanation commands in that dataset's page. Check progress with:

```bash
python src/scripts/check_partition_progress.py --config FunnyBirds_epito PACO_epito KvasirSEG_epito ISIC2018_epito
```

All four use the existing SAM1 and SAM2 checkpoints for partitions. Paths in the `*_epito.yaml` configs are for EPITO.

To include them in the budget study, name them explicitly:

```bash
python src/hpc/prepare_full_budget_study.py --repo . --study <folder> --datasets FunnyBirds PACO KvasirSEG ISIC2018
```

The default study remains the original six datasets.

## Check your setup

From the repository root, on a GPU node. It only reads files and package information. On the login node, dataset and checkpoint results are still right, but packages can be reported as missing because the environment is built for the GPU nodes:

```bash
python src/scripts/summary_setup.py
```

It prints tables for datasets, checkpoints, and packages, then a **To fix** list with the page to read for each problem.

| Column | Meaning |
| --- | --- |
| `Partitions: READY` | Partition generation can run |
| `XAI: READY` | Explanations can run now |
| `XAI: NO PARTS` | Everything is in place except partitions, so generate them first |
| `BLOCKED` | Something required is missing; see **To fix** |

Useful options:

```bash
python src/scripts/summary_setup.py --config PACO_epito KvasirSEG_epito   # Selected datasets
python src/scripts/summary_setup.py --verify-hash                         # Also check SHA-256; slower
python src/scripts/summary_setup.py --strict                              # Exit 1 unless all are READY for XAI
python src/scripts/summary_setup.py --json results/setup.json             # Also save a report
```

The command exits with 1 if any dataset is `BLOCKED`. For exact partition progress and timing, use `check_partition_progress.py`.

## Check partition quality

Before spending time on XAI for a dataset, [evaluate its partitions](../Partitions/EVALUATE_PARTITIONS.md) against the ground truth. It gives a verdict for each dataset.

## Added datasets

- [x] MS-COCO
- [x] ImageNet
- [x] [PartImageNet](docs/DATASETS/PARTIMAGENET.md)
- [x] Celeb-A-HQ
- [x] CAMO
- [x] [BUSI](docs/DATASETS/BUSI.md)
- [x] [FunnyBirds](docs/DATASETS/FUNNY_BIRD.md)
- [x] [PACO: Parts and Attributes of Common Objects](docs/DATASETS/PACO.md)
- [x] [Kvasir-SEG](docs/DATASETS/KVASIR_SEG.md)
- [x] [ISIC-2018](docs/DATASETS/ISIC_2018.md)
