# Kvasir-SEG

[Kvasir-SEG](https://datasets.simula.no/kvasir-seg) ([repository](https://github.com/DebeshJha/Kvasir-SEG)) provides expert polyp masks for colonoscopy images. Every image contains a polyp, so the question is direct: does the explanation of the `polyps` score land on the polyp?

|              |                                                      |
| ------------ | ---------------------------------------------------- |
| Images       | 1,000 polyp images, 332×487 to 1920×1072             |
| Ground truth | One binary polyp mask per image, plus bounding boxes |
| Model | [Kvasir v2 ViT](../MODELS/KVASIR_VIT.md) |
| Download | 46 MB |
| Terms        | Research and education only; cite the dataset paper  |

This is **not** [KvasirCapsule-SEG](https://datasets.simula.no/kvasir-capsule-seg/), a separate 55-image capsule endoscopy set.

## 1. Download the dataset

```bash
python src/scripts/download_datasets.py --dataset kvasirseg --assets dataset
```

The dataset goes to `/beegfs/home/mrashid/datasets/Kvasir-SEG/{images,masks}`.

Simula's server sends an incomplete TLS certificate chain, which Python rejects. The script then retries that one file without certificate verification, and only keeps it if it matches the pinned SHA-256. You will see a message when this happens.

## 2. Models

- [Kvasir v2 ViT](../MODELS/KVASIR_VIT.md)

## 3. Next steps

1. [Generate partitions](../Partitions/DATASETS/KVASIR_SEG.md)
2. [Run explanations](../XAI/DATASETS/KVASIR_SEG.md)

## Limits on what you can claim

- **Train/explain overlap.** Kvasir-SEG's images come from Kvasir v2's polyp class, which is this model's training data, and the card gives no split. Assume the model has seen most of these images. Attributions still describe this model's behaviour, but say so in any paper, and do not report them as held-out performance.
- Black borders cover about 15% of the average image, and `black` is one of the replacement backgrounds. Masked regions can therefore look like the border. The default `--bg-type full` averages this over five backgrounds; avoid `black` alone here.
- Masks are JPEG files, so the loader thresholds at 128. Using `> 0` would add compression noise at the edges.
- Research diagnostics only, not clinical evidence.
