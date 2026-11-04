# EPITO: CAMO Train partitions

Use the existing SAM GPU environment from [EPITO.md](../EPITO.md#1-start-a-gpu-session).
Sync `generate_camo_partitions.py` and `src/configs/CAMO_epito.yaml` to EPITO,
along with the shared generation scripts if they are not current.

The config scans the folder containing your example image:

```text
/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/data/data_full/CAMO-V.1.0/Images/Train
```

Only Train images are included. Original filename stems such as
`camourflage_00001` are preserved. No detection annotations are needed.

## Check paths

```bash
python src/scripts/generate_camo_partitions.py --limit 2 --dry-run
```

## SAM1 (ViT-B)

```bash
python src/scripts/generate_camo_partitions.py \
  --config CAMO_epito \
  --model sam1 --sam1-variant vit_b \
  --checkpoint checkpoints/sam1/sam_vit_b_01ec64.pth \
  --device cuda --max-regions 62 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact \
  --limit 1000
```

## SAM2 - tiny

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_camo_partitions.py \
  --config CAMO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact \
  --limit 2
```

## Preview

```bash
python src/scripts/view_partitions.py \
  --config CAMO_epito \
  --limit 5 \
  --methods sam compact
```

## Resume after CUDA out-of-memory

Shared inference now halves the point batch on a CUDA out-of-memory error
(`64 → 32 → 16 → ... → 1`), clears failed cached image features, and retries the
current image. The smaller batch remains in use for subsequent images. Retry
messages go to the run log. Other errors still stop the run.

Sync the updated `generate_sam_partitions.py` and `generate_coco_partitions.py`,
then repeat your exact command. Completed images are retained and skipped;
do not add `--overwrite`. Requested settings remain unchanged for resume, while
JSON records the actual `effective_points_per_batch` used for each new image.
If batch size 1 still fails, the error is logged and propagated.

The raw-mask target and final region cap do not cap GPU memory during candidate mask generation and resizing. Explicitly changing `--points-per-batch` changes saved generation settings and can trigger the resume mismatch check, so use the automatic retry when continuing an existing dataset.
