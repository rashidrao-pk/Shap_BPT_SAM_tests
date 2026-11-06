# VGG16 (ImageNet)

| | |
| --- | --- |
| Used with | [ImageNet](../DATASETS/ImageNet.md) |
| Task | Classification, 1000 ImageNet classes |
| Runner | `OC_VGG` |
| Weights | torchvision `IMAGENET1K_V1`, 71.6% top-1 |
| Input | Resize to 256, center crop to 224, ImageNet mean/std |

## Download

torchvision uses its own cache (553 MB). Fill it on a node with internet access:

```bash
python -c "from torchvision.models import vgg16; vgg16(weights='DEFAULT')"
```

## Why this model

It is a plain CNN without residual connections, so it widens the architecture comparison with [ResNet-50](RESNET50.md) and [ViT-B/16](VIT_B16.md).

Usage: [ImageNet XAI](../XAI/DATASETS/IMAGENET.md).
