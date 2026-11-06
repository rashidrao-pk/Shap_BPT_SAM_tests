# CAMO

[CAMO](https://sites.google.com/view/ltnghia/research/camo) ([paper](https://arxiv.org/abs/2105.09451)) contains images of camouflaged objects. Objects blend into the background by design, so it is a hard test for partitions built from image structure.

| | |
| --- | --- |
| Images | 1,250: 1,000 train, 250 test (JPEG) |
| Ground truth | One binary object mask per image (PNG) |
| Model | [ViT-B/16](../MODELS/VIT_B16.md) |

## 1. Download the dataset

The archive is on Google Drive, so download it by hand:

1. Open the [CAMO page](https://sites.google.com/view/ltnghia/research/camo) and use the **Datasets** link to get `CAMO-V.1.0`.
2. Copy it to EPITO and extract it:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/data/data_full
unzip -q CAMO-V.1.0.zip
```

The layout on EPITO is:

```text
CAMO-V.1.0/
├── Images/
│   ├── Train/camourflage_00001.jpg
│   └── Test/camourflage_00012.jpg
├── GT/camourflage_00001.png
└── Overlaid_Mask/
```

`GT/` is one flat folder with all 1,250 masks. Masks match images by filename, so train and test images use the same folder and `ground_truth.split` has no effect. All 1,250 images have a mask.

The config explains `Images/Train`. To use the test images, change `data.image_dir` to `Images/Test`. The partition folder name does not include the split, so also use a separate `output.partitions` root.

Cite the [CAMO paper](https://arxiv.org/abs/2105.09451) when using this dataset.

## 2. Models

- [ViT-B/16](../MODELS/VIT_B16.md)

## Next steps

1. [Generate partitions](../Partitions/DATASETS/CAMO_EPITO.md)
2. [Run explanations](../XAI/DATASETS/CAMO.md)
