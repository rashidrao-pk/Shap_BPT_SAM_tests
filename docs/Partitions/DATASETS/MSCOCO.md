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
  --max-regions 254 \
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
  --max-regions 254 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact \
  --limit 1000
```

Remove `--limit 1000` for all 5,000 val2017 images. Repeat the same command to resume: completed images are skipped.

### Specific images

Add `--image-id ID [ID ...]` to generate only those images, for example to redo or inspect one:

```bash
python src/scripts/generate_coco_partitions.py --config MSCOCO_epito_sam1 --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" --sam1-variant vit_b --device cuda \
  --generate_at_least_n_masks 200 --save-partitions sam compact --image-id 76211 28285 57725 113867 11615
```

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda --generate_at_least_n_masks 200 --save-partitions sam compact --image-id 76211 28285 57725 113867 11615
```

COCO IDs work padded or not (`76211` = `000000076211`); other datasets use the file stem, e.g. `--image-id "benign (12)"`. Unknown IDs stop the run before SAM loads. The output folder and resume settings are the same as for a full run, so the images join the normal folder, and a later full run skips them. Completed images are skipped unless `--overwrite` is given. Every `generate_*_partitions.py` has this flag.

### Choose the partitions for XAI

| Model       | Folder                                  |
| ----------- | --------------------------------------- |
| SAM-1       | `MSCOCO_sam1_vit_b_254_compact-sam_200` |
| SAM-2 Large | `MSCOCO_sam2_large_254_compact-sam_200` |

`data.mask_dir_final` selects the folder used by XAI. `MSCOCO_epito` currently selects SAM-1.

## Preview the first five completed images

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --limit 5 \
  --methods sam compact
```

- Or preview specific images by COCO ID (one or more, padded or not). An image without partitions is reported and skipped, and the others still render:

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --sam sam2 \
  --image-id 76211 28285 57725 113867 11615 \
  --methods sam compact
```

- Choose the model with `--sam sam1` or `--sam sam2`. These read `data.mask_dir_final_sam1` / `_sam2` from the config. Without `--sam`, the preview uses `data.mask_dir_final` (SAM-1 for `MSCOCO_epito`). Use `--partition-folder NAME` for any other folder, for example an older `MSCOCO_sam1_vit_b_62_compact-sam_200`:

```bash
python src/scripts/view_partitions.py --config MSCOCO_epito --image-id 76211 --methods sam compact --sam sam2
```

Previews are saved under `previews/<partition folder>/`, so SAM-1 and SAM-2 previews of the same image do not overwrite each other.

## Progress

```bash
python src/scripts/check_partition_progress.py --config MSCOCO_epito --limit 1000
```

Next: [MS-COCO XAI](../../XAI/DATASETS/MSCOCO.md).
