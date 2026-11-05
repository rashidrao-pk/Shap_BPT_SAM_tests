# EPITO: ImageNet partition generation

Use the existing SAM environment and GPU session from [EPITO.md](EPITO.md#1-start-a-gpu-session).
Sync the new `generate_imagenet_partitions.py`, updated `generate_coco_partitions.py`,
and `src/configs/ImageNet_epito.yaml` to EPITO first.

The config reads images from:

```text
/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/data/data_full/ImageNet/ImageNet/ILSVRC2012_img_val
```

No annotations are needed. Images are discovered recursively in sorted relative-path
order; `.JPEG` and other common image extensions are accepted. Only one image is
decoded at a time. Duplicate filename stems are rejected to prevent overwrites.

## Check paths

```bash
python src/scripts/generate_imagenet_partitions.py --limit 2 --dry-run
```

## Generate two images

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_imagenet_partitions.py \
  --config ImageNet_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --limit 2
```

## Generate all images

Repeat the command without `--limit 2`. Completed images are skipped when settings
match. The text tqdm bar shows progress; counts, tuning attempts, and timings go
to the matching log:

```text
/beegfs/home/mrashid/datasets/SAM_BPT/ImageNet_sam2_tiny_62_refined_200/
/beegfs/home/mrashid/datasets/SAM_BPT/ImageNet_sam2_tiny_62_refined_200.log
```

For each image, the folder contains:

```text
ILSVRC2012_val_00000001_sam.npy
ILSVRC2012_val_00000001_refined.npy
ILSVRC2012_val_00000001_results.json
```

The JSON records the source image, settings, mask/region counts, tuning history,
and timing. The target of 200 raw masks is a bounded search; final maps have at
most 62 regions. Omit the tuning flag for a single pass (`_singlepass` suffix).
Use `--overwrite` to regenerate completed images.

For Large, pass both `--checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt"`
and `--sam2-config configs/sam2.1/sam2.1_hiera_l.yaml`. Its folder and log use
`large` instead of `tiny`.

These outputs prepare partitions only. The COCO XAI runner expects COCO image IDs
and annotations; ImageNet XAI integration is a separate step.

## Preview generated ImageNet partitions

Set `data.mask_dir_final` in `src/configs/ImageNet_epito.yaml` to the folder you
generated, for example `ImageNet_sam2_large_62_refined_200`.
The viewer reads image, partition, and output paths from that config.

Preview the first five completed images:

```bash
python src/scripts/view_partitions.py \
  --config ImageNet_epito \
  --limit 5 \
  --methods sam refined
```

Or preview one image by its validation number or full filename stem:

```bash
python src/scripts/view_partitions.py \
  --config ImageNet_epito \
  --image-id 1 \
  --methods sam compact boosted filled boosted_filled refined
```

Use only the stages you saved. `--image-id 1` expands to
`ILSVRC2012_val_00000001`. Batch previews skip images that do not yet have every
requested stage and a results JSON, so they can inspect completed work during
generation.

PNG previews and matching statistics JSON files go to:

```text
<output.evaluations>/previews/<mask_dir_final>/ILSVRC2012_val_00000001.png
```

No GPU or SAM model is needed to view saved partitions.

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
