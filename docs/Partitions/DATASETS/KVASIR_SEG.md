## Generate Partitions for `Kvasir-SEG` Dataset

- [Connect first](../EPITO.md#1-start-a-gpu-session)
- [Download the dataset](../../DATASETS/KVASIR_SEG.md)
- Download checkpoints: [SAM-1](../../MODELS/SAM1.md), [SAM-2](../../MODELS/SAM2.md)

Run from the repository root on a GPU node. Ground-truth masks are never used to generate partitions.

### Check paths

```bash
python src/scripts/generate_kvasirseg_partitions.py \
  --config KvasirSEG_epito --limit 2 --dry-run
```

### SAM-1

`--compact-strategy greedy split` saves both compact selections from one SAM run: `<id>_compact_greedy.npy` and `<id>_compact_split.npy` in the same folder. `data.compact_strategy` in the config picks the one used by XAI (default `greedy`); see [MSCOCO.md](MSCOCO.md#compare-compact-strategies-greedy-vs-mso).

```bash
python src/scripts/generate_kvasirseg_partitions.py \
  --config KvasirSEG_epito \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda \
  --max-regions 254 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact --compact-strategy greedy split \
  --limit 2
```

### SAM-2

```bash
python src/scripts/generate_kvasirseg_partitions.py \
  --config KvasirSEG_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --max-regions 254 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact --compact-strategy greedy split \
  --limit 2
```

## MedSAM

```bash
python src/scripts/generate_kvasirseg_partitions.py \
  --config KvasirSEG_epito \
  --model medsam \
  --device cuda \
  --max-regions 254 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact --compact-strategy greedy split \
  --limit 2

```

Remove `--limit 2` to process all 1,000 images. Repeat the same command to resume: completed images are skipped.

Keep `--save-partitions sam compact`. Saving other stages changes the folder name, so it would no longer match the config or the budget study.

### Choose the partitions for XAI

| Model | Folder                                    |
| ----- | ----------------------------------------- |
| SAM-1 | `KvasirSEG_sam1_vit_b_254_compact-sam_50` |
| SAM-2 | `KvasirSEG_sam2_large_254_compact-sam_50` |

The config selects **SAM-1** (`data.mask_dir_final`), unlike most datasets. To explain SAM-2 partitions, set `data.mask_dir_final` to the SAM-2 folder. For previews, `--sam sam1` or `--sam sam2` picks the folder; without it, the viewer uses the SAM-1 folder.

## Preview the first five completed images

```bash
python src/scripts/view_partitions.py \
  --config KvasirSEG_epito \
  --sam sam2 \
  --limit 5 \
  --methods sam compact_greedy compact_split \
  --show-gt
```

```bash
python src/scripts/view_partitions.py --config KvasirSEG_epito --sam medsam --limit 2 \
  --methods sam compact_greedy compact_split --show-gt
```

- Or preview one image by its filename stem:

```bash
python src/scripts/view_partitions.py \
  --config KvasirSEG_epito \
  --image-id cju0qkwl35piu0993l0dewei2 \
  --sam sam2 \
  --methods sam compact_greedy compact_split \
  --show-gt
```

The preview shows the binary polyp mask.

## Progress

```bash
python src/scripts/check_partition_progress.py --config KvasirSEG_epito
```

The expected total is 1,000.

Next: [Kvasir-SEG XAI](../../XAI/DATASETS/KVASIR_SEG.md).

### Medical priors: MedSAM and MedicoSAM

Same settings, with a medical SAM instead. Setup, downloads and how they work: [MEDICAL_SEGMENTERS.md](../SAM/MEDICAL_SEGMENTERS.md).

```bash
python src/scripts/generate_kvasirseg_partitions.py --config KvasirSEG_epito --model medsam --device cuda \
  --max-regions 254 --generate_at_least_n_masks 50 --save-partitions sam compact --compact-strategy greedy split --limit 1000
python src/scripts/generate_kvasirseg_partitions.py --config KvasirSEG_epito --model medicosam --device cuda \
  --max-regions 254 --generate_at_least_n_masks 50 --save-partitions sam compact --compact-strategy greedy split --limit 1000
python src/scripts/view_partitions.py --config KvasirSEG_epito --sam medsam --limit 5 --methods sam compact_greedy compact_split
```
