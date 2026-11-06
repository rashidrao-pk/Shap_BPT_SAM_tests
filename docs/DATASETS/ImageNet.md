# ImageNet and ImageNet-S

The images come from the ImageNet-1k validation set. [ImageNet-S](https://github.com/LUSSeg/ImageNet-S) adds semantic masks for part of it. The repository uses ImageNetS50, following the reference notebook.

| | |
| --- | --- |
| Images | 50,000 validation images (6.3 GB) |
| Ground truth | Masks for 752 images (ImageNetS50), 4,097 (S300), 12,419 (S919) |
| Models | [ResNet-50](../MODELS/RESNET50.md), [VGG16](../MODELS/VGG16.md), [ViT-B/16](../MODELS/VIT_B16.md) |

## 1. Download the images

ImageNet needs an account:

1. Register at [image-net.org](https://image-net.org/download-images) and accept the terms, which limit use to non-commercial research.
2. Download `ILSVRC2012_img_val.tar` from the 2012 challenge page.
3. Extract it:

```bash
ROOT=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/data/data_full/ImageNet
mkdir -p "$ROOT/ImageNet/ILSVRC2012_img_val"
tar -xf ILSVRC2012_img_val.tar -C "$ROOT/ImageNet/ILSVRC2012_img_val"
```

The training set is not needed. The authors' `data_preparation.sh` is not needed either, because this repository reads the original validation folder and matches masks by filename.

## 2. Download the ImageNet-S annotations

No account is needed:

```bash
ROOT=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/data/data_full/ImageNet
S="$ROOT/ImageNetS"
mkdir -p "$S/ILSVRC2012_img_val_S"
curl --fail --location -o "$S/ImageNetS50.zip" \
  https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS50-a0fe9d82231f9bc4787ee76e304dfa51.zip
unzip -q "$S/ImageNetS50.zip" -d "$S/ILSVRC2012_img_val_S"
curl --fail --location -o "$S/ImageNetS_categories_im50.txt" \
  https://raw.githubusercontent.com/LUSSeg/ImageNet-S/main/data/categories/ImageNetS_categories_im50.txt
```

For larger variants, see the [annotation script](https://github.com/LUSSeg/ImageNet-S/blob/main/datapreparation/datapreparation_anno.sh), then set `ground_truth.variant` to `300` or `919`.

The layout on EPITO is:

```text
data_full/ImageNet/
├── ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG
├── ImageNetS/
│   ├── ImageNetS_categories_im50.txt
│   └── ILSVRC2012_img_val_S/ImageNetS50/
│       ├── validation-segmentation/n01443537/ILSVRC2012_val_00004677.png
│       └── train-semi-segmentation/
└── imagenet_class_index.json
```

`ImageNet_selected/` (593 images) also exists on EPITO. The config does not use it.

Masks are RGB, with class ID `R + 256 * G`. Do not read them as grayscale.

## 3. Models

- [ResNet-50](../MODELS/RESNET50.md)
- [VGG16](../MODELS/VGG16.md)
- [ViT-B/16](../MODELS/VIT_B16.md)

## Next steps

1. [Generate partitions](../Partitions/DATASETS/ImageNet.md)
2. [Run explanations](../XAI/DATASETS/IMAGENET.md)
