## Generate Partitions for `RefCOCO` Dataset

- [Connect first](../EPITO.md#1-start-a-gpu-session)
- [Download the dataset](../../DATASETS/REFCOCO.md)
- Download checkpoints: [SAM-1](../../MODELS/SAM1.md), [SAM-2](../../MODELS/SAM2.md)

Run from the repository root on a GPU node. Partitions are made once per image (1,500), and every expression of that image uses them. Ground truth and expressions are never used to generate partitions.

### Check paths

```bash
python src/scripts/generate_refcoco_partitions.py \
  --config RefCOCO_epito --limit 2 --dry-run
```

### SAM-2

```bash
python src/scripts/generate_refcoco_partitions.py \
  --config RefCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact \
  --limit 2
```

### SAM-1 (optional, to compare with SAM-2)

```bash
python src/scripts/generate_refcoco_partitions.py \
  --config RefCOCO_epito \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact \
  --limit 2
```

Remove `--limit 2` for all 1,500 images. Repeat the same command to resume. Keep `--save-partitions sam compact`, so the folder matches the config. `--generate_at_least_n_masks 200` matches MS-COCO, since these are COCO photos.

These are COCO `train2014` photos, so the existing MS-COCO partitions (`val2017`) cannot be reused.

### Choose the partitions for XAI

| Model | Folder |
| --- | --- |
| SAM-1 | `RefCOCO_sam1_vit_b_62_compact-sam_200` |
| SAM-2 | `RefCOCO_sam2_tiny_62_compact-sam_200` |

## Preview

```bash
python src/scripts/view_partitions.py \
  --config RefCOCO_epito --limit 5 --methods sam compact --show-gt
```

The preview shows every referred object of the image.

## Progress

```bash
python src/scripts/check_partition_progress.py --config RefCOCO_epito
```

The expected total is 1,500 images.

## Evaluate against GT

```bash
python src/scripts/evaluate_partitions.py --config RefCOCO_epito
```

Each referred object is scored on its own. Objects no expression refers to are not GT. See [Evaluate partitions](../EVALUATE_PARTITIONS.md).

Next: [RefCOCO XAI](../../XAI/DATASETS/REFCOCO.md).
