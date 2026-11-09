## Generate Partitions for `DOTA` Dataset

- [Connect first](../EPITO.md#1-start-a-gpu-session)
- [Download the dataset and cut tiles](../../DATASETS/DOTA.md)
- Download checkpoints: [SAM-1](../../MODELS/SAM1.md), [SAM-2](../../MODELS/SAM2.md)

Run from the repository root on a GPU node. One tile is one image. Ground truth is never used to generate partitions.

### Check paths

```bash
python src/scripts/generate_dota_partitions.py \
  --config DOTA_epito --limit 2 --dry-run
```

### SAM-2

```bash
python src/scripts/generate_dota_partitions.py \
  --config DOTA_epito \
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
python src/scripts/generate_dota_partitions.py \
  --config DOTA_epito \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact \
  --limit 2
```

Remove `--limit 2` to process every tile. Repeat the same command to resume: completed tiles are skipped.

Keep `--save-partitions sam compact`. Without it the folder becomes `..._refined-sam_singlepass`, which the config and budget study do not use.

`--generate_at_least_n_masks 200` matches MS-COCO: tiles hold many small objects. Partitions keep at most 62 regions (plus 0 for unassigned pixels), so a tile with more objects than that merges some of them. That is expected and is what the partition evaluation measures.

### Choose the partitions for XAI

| Model | Folder |
| --- | --- |
| SAM-1 | `DOTA_sam1_vit_b_62_compact-sam_200` |
| SAM-2 | `DOTA_sam2_tiny_62_compact-sam_200` |

The config selects SAM-2. To explain SAM-1 partitions, set `data.mask_dir_final` to the SAM-1 folder.

## Preview

```bash
python src/scripts/view_partitions.py \
  --config DOTA_epito \
  --limit 5 \
  --methods sam compact \
  --show-gt
```

The preview colours each counted object by class.

## Progress

```bash
python src/scripts/check_partition_progress.py --config DOTA_epito
```

The expected total is the number of selected tiles (300 by default). The folder name does not include the split, tile size, stride or sample, so use a separate `output.partitions` root when changing any of them.

## Evaluate against GT

```bash
python src/scripts/evaluate_partitions.py --config DOTA_epito
python src/scripts/evaluate_partitions.py --config DOTA_epito --all-sam   # SAM-1 and SAM-2 folders
```

Each counted object is scored on its own, so overlapping boxes (a ship in a harbor) do not cut each other. Ignored regions are left out. See [Evaluate partitions](../EVALUATE_PARTITIONS.md) for the verdict and the per-class and per-size diagnosis.

Next: [DOTA XAI](../../XAI/DATASETS/DOTA.md).
