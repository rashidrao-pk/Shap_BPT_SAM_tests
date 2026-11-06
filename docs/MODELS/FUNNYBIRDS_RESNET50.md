# FunnyBirds ResNet-50

| | |
| --- | --- |
| Used with | [FunnyBirds](../DATASETS/FUNNY_BIRD.md) |
| Task | Classification, 50 bird classes |
| Runner | `OC_FunnyBirds` |
| Input | Resize to 256×256, scaled to 0–1, **no mean/std** |
| Checkpoint | `checkpoints/funnybirds/resnet50_final_0_checkpoint_best.pth.tar` (189 MB) |

## Download

```bash
python src/scripts/download_datasets.py --dataset funnybirds --assets model
```

The script checks the SHA-256.

## Why this model

- It is the [authors' checkpoint](https://github.com/visinf/funnybirds-framework), trained on FunnyBirds train. An ImageNet classifier has no FunnyBirds classes, so a part mask would say nothing about its decision.
- It got 80 of 80 sampled test images right. The authors report 99.8% accuracy.
- All explanation methods ran on a real image, including LRP and GradCAM.

## Limits

- Do not use `OC_ResNet` here. With ImageNet preprocessing the model got 7 of 40 sampled images right.
- The authors also publish VGG16 and ViT-B/16 checkpoints. They are not wired in.
