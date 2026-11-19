# Kvasir v2 ViT

| | |
| --- | --- |
| Used with | [Kvasir-SEG](../DATASETS/KVASIR_SEG.md) |
| Task | Classification, 8 Kvasir v2 classes; `polyps` is index 6 |
| Runner | `OC_KvasirSEG` |
| Input | Direct resize to 224×224, mean/std 0.5 |
| Checkpoint | `checkpoints/kvasir_vit` (343 MB) |

## Download

```bash
python src/scripts/download_datasets.py --dataset kvasirseg --assets model
```

Files are pinned to one Hugging Face commit and checked by SHA-256.

## Why this model

[mmuratarat/kvasir-v2-classifier](https://huggingface.co/mmuratarat/kvasir-v2-classifier) is the most practical public classifier: it loads with plain Transformers, is ungated, and has a `polyps` class. It predicted `polyps` for 199 of 200 sampled Kvasir-SEG images.

[ACPolyp-v1](https://huggingface.co/VishnuRuthlessCoder/acpolyp-v1) reports a scene-grouped split, a better leakage guard, but it is gated and needs its own package. Consider it if leakage matters for a claim.

## Limits

- **It likely trained on these images.** Kvasir-SEG comes from Kvasir v2's polyp class, and the model card gives no split.
- LRP is not available for this architecture. GradCAM works.
- Kvasir data is for research and education only.

## If loading fails

The checkpoint is a `pytorch_model.bin` file. Recent Transformers versions may refuse this format on older PyTorch. If you see a `torch.load` safety error on EPITO, do not replace the site PyTorch; use a Transformers version that works with it.
