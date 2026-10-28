# EPITO: CelebA-HQ partitions

Use your existing SAM environment and GPU allocation from
[EPITO.md](EPITO.md#1-start-a-gpu-session). Sync the new script and
`src/configs/CelebAHQ_epito.yaml` to EPITO, together with the shared generation
scripts if they are not up to date.

The config reads images from:

```text
/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/data/data_full/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img
```

No face annotations are required. One image is loaded at a time; original
filename stems are preserved (`0.jpg` produces `0_sam.npy`, without COCO padding).
Images are processed in sorted relative-path order, not numeric filename order.

## Check inputs

```bash
python src/scripts/generate_celebahq_partitions.py --limit 2 --dry-run
```

## Generate two images with SAM2.1 Large

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_celebahq_partitions.py \
  --config CelebAHQ_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact refined \
  --limit 2
```

Remove `--limit 2` for the full dataset. Resume with the same command; completed
images with matching settings are skipped. Default outputs and log are:

```text
/beegfs/home/mrashid/datasets/SAM_BPT/CelebAHQ_sam2_large_62_refined_200/
/beegfs/home/mrashid/datasets/SAM_BPT/CelebAHQ_sam2_large_62_refined_200.log
```

The terminal shows text tqdm; the log records mask counts, saved stage names,
label counts, and timing. Each image has a `*_results.json` with exact saved paths.

Select only the stages you need with `--save-partitions`. Without that option,
only `sam` and `refined` are saved. These two stages are capped to 62 labels;
notebook stages retain their original behavior. The raw mask target of 200 is
a bounded threshold search, not a guarantee. See [stage details](EPITO.md#choose-which-partition-stages-to-save).

For Tiny, use `sam2.1_hiera_tiny.pt` with `sam2.1_hiera_t.yaml`, and set
`data.mask_dir_final` to `CelebAHQ_sam2_tiny_62_refined_200` before previewing.

## Preview saved partitions

```bash
python src/scripts/view_partitions.py \
  --config CelebAHQ_epito \
  --limit 5 \
  --methods sam compact boosted filled boosted_filled refined
```

For one image, replace `--limit 5` with `--image-id 0` (or another actual filename
stem). Use only stages that were saved. Previews and statistics JSON files are
written under the configured output directory in
`previews/CelebAHQ_sam2_large_62_refined_200/`. No GPU is needed for viewing.

This prepares partition maps only; it does not add CelebA-HQ support to the COCO
XAI runner or assign semantic face-part names to the generated labels.
