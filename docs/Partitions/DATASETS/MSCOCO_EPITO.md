## Generate Partitions for `MS-COCO` Dataset

- [Connect first](../EPITO.md#1-start-a-gpu-session)
- [Download the dataset](../../DATASETS/MS_COCO.md)
- Download checkpoints: [SAM-1](../../MODELS/SAM1.md), [SAM-2](../../MODELS/SAM2.md)

### Check paths

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito --limit 2 --dry-run
```

### SAM-1

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito_sam1 \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact \
  --limit 1000
```

### SAM-2

#### Large Model

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact \
  --limit 1000
```

Remove `--limit 1000` for all 5,000 val2017 images. Repeat the same command to resume: completed images are skipped.

### Choose the partitions for XAI

| Model | Folder |
| --- | --- |
| SAM-1 | `MSCOCO_sam1_vit_b_62_compact-sam_200` |
| SAM-2 Large | `MSCOCO_sam2_large_62_compact-sam_200` |

`data.mask_dir_final` selects the folder used by XAI. `MSCOCO_epito` currently selects SAM-1.

## Preview the first five completed images

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --limit 5 \
  --methods sam compact
```

- Or preview one image by its COCO ID:

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --image-id 139 \
  --methods sam compact
```

## Progress

```bash
python src/scripts/check_partition_progress.py --config MSCOCO_epito --limit 1000
```

Next: [MS-COCO XAI](../../XAI/DATASETS/MSCOCO.md).
