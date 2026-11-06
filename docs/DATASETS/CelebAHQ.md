# CelebA-HQ

[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) provides face images with manually annotated part masks, so an attribute such as `Smiling` can be compared with the matching face parts.

| | |
| --- | --- |
| Images | 30,000 faces |
| Ground truth | Per-part PNG masks, at 512×512 |
| Model | [CelebA CNN](../MODELS/CELEBA_CNN.md) |
| Terms | Non-commercial research only |

## 1. Download the dataset

The archive is on Google Drive, so download it by hand from the [dataset page](https://github.com/switchablenorms/CelebAMask-HQ#celebamask-hq-dataset-downloads). Copy it to EPITO and extract it under `data/data_full/CelebAMask-HQ/`.

The layout on EPITO is:

```text
CelebAMask-HQ/CelebAMask-HQ/
├── CelebA-HQ-img/0.jpg
├── CelebAMask-HQ-mask-anno/0/00000_hair.png
└── CelebAMask-HQ-attribute-anno.txt
```

Masks are grouped by `image_id // 2000`, so image 4500 is in folder `2`.

**Use the full mask folder on EPITO.** `data_full/CelebAMask-HQ/CelebAMask-HQ-mask-anno` holds folders 0 to 14, for all 30,000 images, and is the folder `CelebAHQ_epito.yaml` uses. The nested `CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno` holds only folder `0`, stopping at image 900.

## 2. Models

- [CelebA CNN](../MODELS/CELEBA_CNN.md)

## Next steps

1. [Generate partitions](../Partitions/DATASETS/CelebAHQ_EPITO.md)
2. [Run explanations](../XAI/DATASETS/CELEBA.md)
