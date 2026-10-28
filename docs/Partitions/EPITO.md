# EPITO: generate MS-COCO partitions

Requires the completed [one-time setup](../full/EPITO_SETUP.md).
Update the code on EPITO before running these commands, including the new
`generate_coco_partitions.py`, the updated `generate_sam_partitions.py`,
`run_yolo_full.py`, and `src/configs/MSCOCO_epito.yaml`.

## 1. Start a GPU session

```bash
ssh epito-mercurio
srun -p epito --gres=gpu:a100:1 -J "ShapBPT" --time=06:00:00 --pty bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages
export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
cd "$PROJECT_DIR"
```

## 2. Check inputs

The config supplies image, annotation, and partition output paths. This command
checks input paths and lists the first two images without loading SAM:

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito --limit 2 --dry-run
```

The configured output folder is:

```text
/beegfs/home/mrashid/datasets/SAM_BPT/sam2_62_refined/
```

Your previous `300_boosted_filled` folder is preserved.

## 3. Generate two images first

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2

python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62 \
  --limit 2
```

Images are selected by ascending COCO image ID. SAM loads once for the batch.
Each completed image produces:

- `<image_id>_sam.npy`: overlapping masks converted to labels, capped to the region budget.
- `<image_id>_refined.npy`: filled and refined partition map, capped to 62 regions.
- `<image_id>_results.json`: model, settings, counts, and generation timings.

The filenames match the XAI loader. Empty SAM output produces a single-region
map. Raw mask stacks and previews are omitted to reduce dataset storage.

## 4. Generate the full dataset

### SAM-1

```bash

python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito-sam1 \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda \
  --max-regions 62 \
  --limit 2
```

### SAM-2

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62
```



The terminal shows only a text tqdm bar with progress, ETA, and the latest
mask count, region count, and image time. Run details and per-image counts,
SAM/refinement timings, total time, and failures are appended to:

```text
/beegfs/home/mrashid/datasets/SAM_BPT/SAM2_MSCOCO.log
```

To watch the log from another terminal:

```bash
tail -f /beegfs/home/mrashid/datasets/SAM_BPT/SAM2_MSCOCO.log
```

This log path is the default for SAM2 with this config. You can also pass
`--log-file /beegfs/home/mrashid/datasets/SAM_BPT/SAM2_MSCOCO.log` explicitly.
The log stores counts and timings; partition arrays remain in the output folder.

Completed images with matching settings are skipped. If the Slurm allocation
ends, start a new GPU session and repeat this command. Use `--overwrite` to
regenerate outputs. For another model or region budget, change `mask_dir_final`
in the config to a separate folder first. Lower `--points-per-batch` if memory
runs out; keep generation settings consistent when resuming a dataset.

## Optional: tune thresholds toward a minimum mask count

Use the notebook's `generate_at_least_n_masks` approach:

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 400
```

The flag alone also targets 400 raw masks. For each image, lower IoU and
stability thresholds by `0.2` per retry, stopping at the target, three attempts,
or minimum thresholds of `0.1`. Use `--mask-tuning-max-steps 5` to allow more
attempts. Each image starts from your original thresholds and reuses the loaded
model. Retries replace the previous masks; they do not accumulate them.

This is a bounded search, so 400 masks are not guaranteed. Final partition maps
still have at most `--max-regions 62` labels. The log records each attempt;
`*_results.json` includes `mask_tuning_history` and `mask_target_met`.

For an existing dataset, choose a new `mask_dir_final` in the config to keep the
previous outputs, or use `--overwrite` to regenerate them with tuning. The
resume check rejects different generation settings. Without the flag, generation
still uses a single pass. The notebook's separate greedy-selection pipeline is
not added here; the existing partition refinement stays in use.

## 5. Run XAI separately

After the first two images complete, follow [XAI/EPITO.md](../XAI/EPITO.md).
You can run the two-image XAI smoke test before generating all COCO images.

When finished, use `exit` to release the GPU allocation.
