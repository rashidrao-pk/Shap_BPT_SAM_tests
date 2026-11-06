# ViT-B/16 (ImageNet)

| | |
| --- | --- |
| Used with | [ImageNet](../DATASETS/ImageNet.md), [CAMO](../DATASETS/CAMO.md) |
| Task | Classification, 1000 ImageNet classes |
| Runners | `OC_ViT`, `COD_ViT` |
| Weights | torchvision `IMAGENET1K_V1`, 81.1% top-1 |
| Input | Resize to 256, center crop to 224, ImageNet mean/std |

## Download

torchvision uses its own cache (346 MB). Fill it on a node with internet access:

```bash
python -c "from torchvision.models import vit_b_16; vit_b_16(weights='DEFAULT')"
```

## Why this model

- It adds a transformer to the [ResNet-50](RESNET50.md) and [VGG16](VGG16.md) comparison.
- CAMO has no classifier of its own, and this one needs no training.

## Limits

- ImageNet has no "camouflaged" class, so on CAMO the prediction is not about camouflage, and the mask marks the hidden object rather than the predicted class.
- LRP is not available for this architecture.

A classifier trained on CAMO must match torchvision's ViT-B/16: see [CAMO XAI](../XAI/DATASETS/CAMO.md#run).
