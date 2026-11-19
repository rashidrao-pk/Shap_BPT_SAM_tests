# ResNet-50 (ImageNet)

| | |
| --- | --- |
| Used with | [ImageNet](../DATASETS/ImageNet.md), [PartImageNet](../DATASETS/PARTIMAGENET.md), [PACO](../DATASETS/PACO.md) |
| Task | Classification, 1000 ImageNet classes |
| Runners | `OC_ResNet`, `OC_PartImageNet`, `OC_PACO` |
| Weights | torchvision `IMAGENET1K_V2`, 80.9% top-1 |
| Input | Resize to 232, center crop to 224, ImageNet mean/std |
| Checkpoint | `checkpoints/partimagenet/resnet50-11ad3fa6.pth` |

## Download

For PartImageNet and PACO, which load the local file:

```bash
python src/scripts/download_partimagenet.py --assets model
```

For ImageNet, torchvision uses its own cache. Fill it on a node with internet access:

```bash
python -c "from torchvision.models import resnet50; resnet50(weights='DEFAULT')"
```

Both are the same weights. Pass the local file with `--checkpoint <file> --weights IMAGENET1K_V2`.

## Why this model

- It needs no training, and its classes map to ImageNet-S synsets.
- It did not train on COCO, unlike [YOLO](YOLO.md) and [DETR](DETR.md). That makes it the cleaner choice for PACO, whose images come from COCO.
- All explanation methods are available, including LRP and GradCAM.

## Limits

- The center crop hides the image edges from the model. Gradient maps are zero there.
- Its classes are not PartImageNet parts or PACO objects, so the explained class can differ from the annotation.
