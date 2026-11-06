# Kvasir-SEG

[Kvasir-SEG](https://datasets.simula.no/kvasir-seg) ([repository](https://github.com/DebeshJha/Kvasir-SEG)) provides expert polyp masks for colonoscopy images. Every image contains a polyp, so the question is direct: does the explanation of the `polyps` score land on the polyp?

| | |
| --- | --- |
| Images | 1,000 polyp images, 332×487 to 1920×1072 |
| Ground truth | One binary polyp mask per image, plus bounding boxes |
| Classifier | ViT-B/16 fine-tuned on Kvasir v2, 8 classes |
| Download | 46 MB dataset, 343 MB checkpoint |
| Terms | Research and education only; cite the dataset paper |

This is **not** [KvasirCapsule-SEG](https://datasets.simula.no/kvasir-capsule-seg/), a separate 55-image capsule endoscopy set.

## 1. Download the dataset and checkpoint

```bash
python src/scripts/download_datasets.py --dataset kvasirseg
```

The dataset goes to `/beegfs/home/mrashid/datasets/Kvasir-SEG/{images,masks}` and the checkpoint to `checkpoints/kvasir_vit/`.

Simula's server sends an incomplete TLS certificate chain, which Python rejects. The script then retries that one file without certificate verification, and only keeps it if it matches the pinned SHA-256. You will see a message when this happens.

## 2. Why this checkpoint

[mmuratarat/kvasir-v2-classifier](https://huggingface.co/mmuratarat/kvasir-v2-classifier) is the most practical public classifier: it loads with plain Transformers, is ungated, and has a `polyps` class. On 200 sampled Kvasir-SEG images it predicted `polyps` for 199. The download is pinned to one commit.

[ACPolyp-v1](https://huggingface.co/VishnuRuthlessCoder/acpolyp-v1) reports a scene-grouped split, which is a better leakage guard, but it is gated and needs its own package. Consider it if the leakage below matters for a claim.

## 3. Generate partitions

```bash
python src/scripts/generate_kvasirseg_partitions.py \
  --config KvasirSEG_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda --max-regions 62 --generate_at_least_n_masks 50 \
  --save-partitions sam compact
```

## 4. Explain

```bash
python src/scripts/explanations/OC_KvasirSEG.py \
  --config KvasirSEG_epito --target-class 6 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --explain_all
```

`--target-class 6` fixes the explained class to `polyps`, so every map answers the same question as the mask. Without it, the runner explains the top prediction, which can be another class.

Results include pointing accuracy and best IoU against the polyp mask. LRP is reported as unsupported for this ViT, with no substitute map.

## Limits on what you can claim

- **Train/explain overlap.** Kvasir-SEG's images come from Kvasir v2's polyp class, which is this model's training data, and the card gives no split. Assume the model has seen most of these images. Attributions still describe this model's behaviour, but say so in any paper, and do not report them as held-out performance.
- Black borders cover about 15% of the average image, and `black` is one of the replacement backgrounds. Masked regions can therefore look like the border. The default `--bg-type full` averages this over five backgrounds; avoid `black` alone here.
- Masks are JPEG files, so the loader thresholds at 128. Using `> 0` would add compression noise at the edges.
- Research diagnostics only, not clinical evidence.

## If checkpoint loading fails

The checkpoint is a `pytorch_model.bin` file. Recent Transformers versions may refuse this format on older PyTorch. If you see a `torch.load` safety error on EPITO, do not replace the site PyTorch; use a Transformers version that works with it.
