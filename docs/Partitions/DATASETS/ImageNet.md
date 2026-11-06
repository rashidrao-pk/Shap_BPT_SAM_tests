## Generate Partitions for `ImagNet` Dataset

- [Connect first](/docs/Partitions/EPITO.md#1-start-a-gpu-session)
- [Download the dataset](../../DATASETS/ImageNet.md)
- Download checkpoints: [SAM-1](../../MODELS/SAM1.md), [SAM-2](../../MODELS/SAM2.md)

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

## Progress for GT-only generation

Use the same selection flag in the progress checker as in generation:

```bash
python src/scripts/check_partition_progress.py --limit 1000 --gt-only
```

`--gt-only` affects ImageNet rows only; other datasets retain their normal selection. ImageNet source images are intersected with the configured ImageNetS annotations **before** applying `--limit`, matching the generator. For example, 752 annotated source images with 752 completed partitions report `752 / 752`, even with `--limit 1000`. Without this flag, the checker compares the first 1,000 unfiltered inputs; `11 / 1000` describes that different subset, not completion of the GT-only run.

To check only ImageNet and avoid errors from other datasets not yet downloaded:

```bash
python src/scripts/check_partition_progress.py \
  --config ImageNet_epito --gt-only --limit 1000
```

The GT variant/root comes from `data.ground_truth` in `ImageNet_epito.yaml`. Missing annotated source images are excluded from the denominator, just as in generation; missing GT roots and empty overlaps are errors. Folder-wide storage/time still includes all partition outputs. This flag is explicit because older partition metadata does not reliably record whether generation used GT-only selection.
