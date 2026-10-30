## MS-COCO

### SAM-1

```bash

python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito-sam1 \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact refined \
  --limit 1000
```

### SAM-2

### Large Model

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact refined \
  --limit 1000
```
