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

For the Tiny/62-region/200-mask commands below, the automatically named output folder is:

```text
/beegfs/home/mrashid/datasets/SAM_BPT/MSCOCO_sam2_tiny_62_refined_200/
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
  --generate_at_least_n_masks 200 \
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
  --generate_at_least_n_masks 200 \
  --limit 2
```

### SAM-2

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62 \
  --save-partitions sam compact \
  --generate_at_least_n_masks 200
```

The terminal shows only a text tqdm bar with progress, ETA, and the latest
mask count, region count, and image time. Run details and per-image counts,
SAM/refinement timings, total time, and failures are appended to:

```text
/beegfs/home/mrashid/datasets/SAM_BPT/MSCOCO_sam2_tiny_62_refined_200.log
```

To watch the log from another terminal:

```bash
tail -f /beegfs/home/mrashid/datasets/SAM_BPT/MSCOCO_sam2_tiny_62_refined_200.log
```

The default log filename matches the generated partition folder name. You can also pass
`--log-file /beegfs/home/mrashid/datasets/SAM_BPT/MSCOCO_sam2_tiny_62_refined_200.log` explicitly.
The log stores counts and timings; partition arrays remain in the output folder.

Completed images with matching settings are skipped. Restart the same command
to resume. Use `--overwrite` to regenerate outputs.

Folders are named automatically as
`<dataset>_<sam>_<model-size>_<max-regions>_refined_<mask-target>`.
For example, Large with 400 masks uses `MSCOCO_sam2_large_62_refined_400`.
Without threshold tuning, the suffix is `singlepass`. Existing folders are not
renamed or moved. Other settings are still checked when resuming.

For XAI, set `data.mask_dir_final` in the config to the generated folder name.
Generation prints that value in the log (and directly during `--dry-run`);
it does not modify your config. The checked-in config selects
`MSCOCO_sam2_tiny_62_refined_200`.

## Choose which partition stages to save

Without extra options, both batch scripts save **sam + refined**, plus the
per-image results JSON. Choose other stages with `--save-partitions`:

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact boosted filled boosted_filled refined \
  --limit 500
```

| Stage / filename suffix | Meaning                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `sam`                   | All final SAM masks converted to labels, capped to the region budget    |
| `compact`               | Greedy selection by new area × predicted IoU, then small-mask overwrite |
| `boosted`               | Notebook SegmentBooster majority-label refinement of compact labels     |
| `filled`                | Distance-limited filling of compact labels                              |
| `boosted_filled`        | Distance-limited filling of boosted labels                              |
| `refined`               | Existing batch connected-component refinement, filling, and region cap  |

`--select-count 400` controls the greedy selection limit.
`--fill-max-distance 3` controls notebook-stage filling in pixels.
Only `sam` and `refined` are capped by `--max-regions`; the notebook stages
preserve their original behavior and can have more labels or unassigned pixels.
Thus `refined` and `boosted` are different algorithms.

The JSON's `saved_partitions` lists exact paths; `partition_label_counts` lists
counts. The log records the selected stages at startup and per-image counts.
Use `--dry-run` to check the requested stages and paths before GPU work.

Existing completed outputs are not upgraded by an already-running job.
To regenerate the same images with a different stage selection, stop any run
writing to that folder and add `--overwrite`. No old stage files are deleted;
the new JSON identifies the files written by that run. Generating extra stages
requires rerunning SAM because batch outputs do not preserve scored raw masks.
The notebook stages require `imageio` as well as the existing plotting dependencies.

To view all saved stages:

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --image-id 139 \
  --methods sam compact boosted filled boosted_filled refined
```

Set `data.mask_dir_final` to the generated folder first. For XAI, include only
saved stages in `evaluation.methods`. Both the COCO and ImageNet generators
support these stage-selection options. The folder/log naming convention stays
the same; the exact stage selection is recorded in JSON.

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

Changing the model size, region budget, or target automatically selects a
separate folder. Without the flag, generation uses a single pass. Existing
partition refinement remains in use.

## 5. Run XAI separately

After the first two images complete, follow [XAI/EPITO.md](../XAI/EPITO.md).
You can run the two-image XAI smoke test before generating all COCO images.

When finished, use `exit` to release the GPU allocation.

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

The raw-mask target and final region cap do not cap GPU memory during candidate
mask generation and resizing. Explicitly changing `--points-per-batch` changes
saved generation settings and can trigger the resume mismatch check, so use the
automatic retry when continuing an existing dataset.
