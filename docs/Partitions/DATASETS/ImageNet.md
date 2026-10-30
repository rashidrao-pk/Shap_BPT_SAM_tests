## Generate Partitions for `ImagNet` Dataset

- [Connect first](/docs/Partitions/EPITO.md#1-start-a-gpu-session)

### SAM-1

```bash

python src/scripts/generate_imagenet_partitions.py \
  --config ImageNet_epito \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 100 \
  --save-partitions sam compact refined \
  --limit 1000
```

### SAM-2

```bash
python src/scripts/generate_imagenet_partitions.py \
  --config ImageNet_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 100 \
  --limit 500
```

```bash
python src/scripts/generate_imagenet_partitions.py \
  --config ImageNet_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 100 \
  --save-partitions sam compact \
  --limit 500
```
