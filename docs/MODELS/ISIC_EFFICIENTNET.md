# ISIC EfficientNet-B0

| | |
| --- | --- |
| Used with | [ISIC 2018](../DATASETS/ISIC_2018.md) |
| Task | Classification, 7 diagnoses (ISIC 2018 Task 3) |
| Runner | `OC_ISIC2018` |
| Input | Direct resize to 320×320, ImageNet mean/std |
| Checkpoint | `checkpoints/isic2018_efficientnet_b0` (16 MB) |
| Licence | CC BY-NC 4.0 |

## Download

```bash
python src/scripts/download_datasets.py --dataset isic2018 --assets model
```

Files are pinned to one Hugging Face commit and checked by SHA-256.

## Why this model

[adeIaide/skin-lesion-7class-efficientnet](https://huggingface.co/adeIaide/skin-lesion-7class-efficientnet) was chosen because:

- Its `config.json` gives class order and normalization, so preprocessing is not guesswork.
- It used a lesion-disjoint split, avoiding the same lesion in train and development data.
- Weights are in `safetensors` format, so loading runs no pickled code.
- Its Task 3 training images share **no image IDs** with Task 1: all 3,694 Task 1 images were compared with 10,015 training IDs.

Reported on its internal split: macro-F1 0.764, balanced accuracy 0.796.

Alternatives: [malemehaute/vit-base-ham10000-skin-lesion](https://huggingface.co/malemehaute/vit-base-ham10000-skin-lesion) documents no preprocessing or label order, and [iamhmh/derm-cnn-ham10000](https://huggingface.co/iamhmh/derm-cnn-ham10000) takes 28×28 input, too coarse for region explanations.

## Limits

- There is no independent validation.
- Mean top-class confidence was 0.46 on 12 sampled Task 1 images, which suggests a domain gap.
- LRP is not available for this architecture. GradCAM uses the last feature block.
- Research diagnostics only, not for medical use.
