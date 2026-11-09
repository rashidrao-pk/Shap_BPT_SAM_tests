# Models

One page per model, with its download command and why it was chosen. Dataset pages link here.

To check that every checkpoint is in place, run `python src/scripts/summary_setup.py`. See [Check your setup](../DATASETS/README.md#check-your-setup).

## Partition generation

| Model | Used by |
| --- | --- |
| [SAM-1](SAM1.md) | Every dataset |
| [SAM-2](SAM2.md) | Every dataset |

## Explained models

| Model | Task | Datasets |
| --- | --- | --- |
| [YOLO11s](YOLO.md) | Detection | MS-COCO |
| [DETR-ResNet-50](DETR.md) | Detection | MS-COCO |
| [ResNet-50](RESNET50.md) | Classification | ImageNet, PartImageNet, PACO |
| [VGG16](VGG16.md) | Classification | ImageNet |
| [ViT-B/16](VIT_B16.md) | Classification | ImageNet, CAMO |
| [CelebA CNN](CELEBA_CNN.md) | Attributes | CelebA-HQ |
| [BUSI SigLIP](BUSI_SIGLIP.md) | Classification | BUSI |
| [FunnyBirds ResNet-50](FUNNYBIRDS_RESNET50.md) | Classification | FunnyBirds |
| [Kvasir v2 ViT](KVASIR_VIT.md) | Classification | Kvasir-SEG |
| [ISIC EfficientNet-B0](ISIC_EFFICIENTNET.md) | Classification | ISIC 2018 |
| [YOLO11s-OBB](YOLO_OBB.md) | Oriented-box detection | DOTA |
| [CLIP ViT-B/32](CLIP.md) | Image–text similarity | RefCOCO |
