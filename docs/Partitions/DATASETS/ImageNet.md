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
  --save-partitions sam compact \
  --gt-only \
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
  --save-partitions sam compact \
  --gt-only \
  --limit 1000
```

## Preview the first five completed images:

```bash
python src/scripts/view_partitions.py \
  --config ImageNet_epito \
  --limit 5 \
  --methods sam compact
```

- Or preview one image by its validation number or full filename stem:

```bash
python src/scripts/view_partitions.py \
  --config ImageNet_epito \
  --image-id 1 \
  --methods sam compact
```
