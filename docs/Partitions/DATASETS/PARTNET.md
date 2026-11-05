# PartImageNet on EPITO

This document covers **PartImageNet (2D images)**. First follow [one-time setup](PARTIMAGENET_SETUP.md) to download the dataset and ResNet50 classifier and set the config paths.

## Generate partitions

## SAM 1

```bash
python src/scripts/generate_partimagenet_partitions.py \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda --max-regions 62 --generate_at_least_n_masks 50 \
  --save-partitions sam compact --limit 2
```

## SAM 2

From the repository root, inside an allocated GPU node with `pt_312_sam` active:

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_partimagenet_partitions.py \
  --config PartImageNet_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda --max-regions 62 --generate_at_least_n_masks 50 \
  --save-partitions sam compact --limit 2
```

Remove `--limit 2` for the entire configured split. The generator reads only images listed in `data.annotation_file`; masks are not used to generate partitions. Text progress, logs, and resume follow the existing dataset generators.

Set `data.mask_dir_final` to the output folder printed by the generator. This example produces `PartImageNet_sam2_tiny_62_compact-sam_50`.

For SAM1, replace the model/checkpoint options with:

```bash
--model sam1 --sam1-variant vit_b \
--checkpoint checkpoints/sam1/sam_vit_b_01ec64.pth
```

Then use the printed SAM1 folder in `mask_dir_final`. Use separate partition output roots or rename folders if comparing different dataset splits with otherwise identical SAM settings: the generated folder name does not include the split.

## Preview

Preview the first five completed images without specifying image IDs:

```bash
python src/scripts/view_partitions.py \
  --config PartImageNet_epito --limit 5 \
  --methods sam compact --show-gt
```

Or select one image explicitly:

```bash
python src/scripts/view_partitions.py \
  --config PartImageNet_epito --image-id YOUR_IMAGE_STEM \
  --methods sam compact --show-gt
```

`--limit` selects up to N completed images in filename order, skipping inputs missing a requested partition stage. Each preview is saved separately under `output.evaluations/previews/<partition-folder>/`. Use either `--limit` or `--image-id`; do not combine them.

The config supplies image, partition, ground-truth, and preview output paths. PartImageNet previews use the same annotation-based image resolver as generation and XAI. Duplicate basenames elsewhere in the extraction do not override an exact annotated path; unresolved ambiguity remains an error. Batch previews are restricted to the selected annotation split. `--gt-class` optionally selects an annotation category name or ID.

Next: [PartImageNet XAI](../../XAI/PARTIMAGENET.md).

## Check progress

```bash
python src/scripts/check_partition_progress.py --limit 1000
```

The default summary includes BUSI and PartImageNet. BUSI excludes annotation PNGs; PartImageNet follows the configured annotation split. Missing dataset/config paths are reported explicitly. BUSI totals stop at the available image count (780 for the full release), even with `--limit 1000`.
