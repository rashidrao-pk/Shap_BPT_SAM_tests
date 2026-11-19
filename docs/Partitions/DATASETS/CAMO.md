# EPITO: CAMO Train partitions

- [Download the dataset](../../DATASETS/CAMO.md)
- Download checkpoints: [SAM-1](../../MODELS/SAM1.md), [SAM-2](../../MODELS/SAM2.md)

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

`--compact-strategy greedy split` saves both compact selections from one SAM run: `<id>_compact_greedy.npy` and `<id>_compact_split.npy` in the same folder. `data.compact_strategy` in the config picks the one used by XAI (`split` in every config); see [MSCOCO.md](MSCOCO.md#compare-compact-strategies-greedy-vs-mso).

```bash
python src/scripts/generate_camo_partitions.py \
  --config CAMO_epito \
  --model sam1 --sam1-variant vit_b \
  --checkpoint checkpoints/sam1/sam_vit_b_01ec64.pth \
  --device cuda --max-regions 254 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact --compact-strategy greedy split \
  --limit 5
```

## SAM2

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_camo_partitions.py \
  --config CAMO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --max-regions 254 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact --compact-strategy greedy split \
  --limit 5
```

## Preview

```bash
python src/scripts/view_partitions.py \
  --config CAMO_epito \
  --sam sam2 \
  --limit 5 \
  --methods sam compact_greedy compact_split
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
