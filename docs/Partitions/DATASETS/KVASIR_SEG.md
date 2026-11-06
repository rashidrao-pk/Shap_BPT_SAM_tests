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

```bash
python src/scripts/generate_kvasirseg_partitions.py \
  --config KvasirSEG_epito \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact \
  --limit 2
```

### SAM-2

```bash
python src/scripts/generate_kvasirseg_partitions.py \
  --config KvasirSEG_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact \
  --limit 2
```

Remove `--limit 2` to process all 1,000 images. Repeat the same command to resume: completed images are skipped.

Keep `--save-partitions sam compact`. Saving other stages changes the folder name, so it would no longer match the config or the budget study.

### Choose the partitions for XAI

| Model | Folder |
| --- | --- |
| SAM-1 | `KvasirSEG_sam1_vit_b_62_compact-sam_50` |
| SAM-2 | `KvasirSEG_sam2_tiny_62_compact-sam_50` |

The config selects SAM-2. To explain SAM-1 partitions, set `data.mask_dir_final` to the SAM-1 folder.

## Preview the first five completed images

```bash
python src/scripts/view_partitions.py \
  --config KvasirSEG_epito \
  --limit 5 \
  --methods sam compact \
  --show-gt
```

- Or preview one image by its filename stem:

```bash
python src/scripts/view_partitions.py \
  --config KvasirSEG_epito \
  --image-id cju0qkwl35piu0993l0dewei2 \
  --methods sam compact \
  --show-gt
```

The preview shows the binary polyp mask.

## Progress

```bash
python src/scripts/check_partition_progress.py --config KvasirSEG_epito
```

The expected total is 1,000.

Next: [Kvasir-SEG XAI](../../XAI/DATASETS/KVASIR_SEG.md).
