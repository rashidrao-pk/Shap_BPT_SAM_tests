# Notebook pipeline scripts

Source: [pipeline-sam.ipynb](../../notebooks/pipeline-sam.ipynb).
All 13 function definitions are available in
[sam_notebook_pipeline.py](../../src/scripts/sam_notebook_pipeline.py).
Importing it does not execute notebook installation cells or load images/models.

## View existing COCO or ImageNet partitions

From the repository root, use the dataset config:

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --image-id 139
```

This reads the image from `data.image_dir` and partitions from
`data.masks_path / data.mask_dir / data.mask_dir_final`. Set `mask_dir_final`
to the dataset you want to inspect (for example,
`MSCOCO_sam2_large_62_refined_200`). COCO IDs are padded to 12 digits.
The preview is saved under the config's `output.dir / output.folder`, in
`previews/<mask_dir_final>/<image_id>.png`, with a matching statistics JSON.
An absolute `output.folder` takes precedence over `output.dir`, as in the XAI runner.

By default the viewer compares `sam` and `refined`. Select other stage suffixes
with `--methods sam boosted boosted_filled` when those files exist.
ImageNet configs need an `output.dir` or `output.folder` too, or an explicit
`--output` argument. Use the full filename stem as `--image-id` for ImageNet.

You can still override individual paths or run without a config:

```bash
python src/scripts/view_partitions.py \
  --image /path/to/000000000139.jpg \
  --partitions /path/to/000000000139_sam.npy /path/to/000000000139_refined.npy \
  --output results/previews/000000000139.png
```

This saves original/label/overlay panels plus a JSON of dimensions, label counts,
and region sizes. It works headlessly and requires no SAM installation. Input
maps must be integer 2D arrays matching the image dimensions. Zero is counted as
a label: batch partitions may use zero as a valid region. Colors are local to
each map; equal colors across stages do not establish region correspondence.
Pick actual generated image IDs; the filenames above are examples.

## Run the notebook stages on one image with SAM2

Inside the existing EPITO GPU environment:

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/run_notebook_pipeline.py \
  --image data/sample_images/truck.jpg \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --generate_at_least_n_masks 200 \
  --select-count 400 \
  --output-dir results/notebook_pipeline/truck
```

For Large, use `sam2.1_hiera_large.pt` and `sam2.1_hiera_l.yaml` together.
The target mask count is bounded by three threshold-tuning attempts by default;
`--mask-tuning-max-steps` changes that bound.

The stages are:

1. Generate raw masks and preserve predicted IoU scores.
2. Greedily select up to `--select-count` masks by uncovered area × IoU; stop when none adds pixels.
3. Build compact labels, with smaller masks overwriting larger masks.
4. Apply the notebook's SegmentBooster majority-label refinement.
5. Fill zero pixels up to `--max-distance` pixels (default `3`).

Saved outputs include `*_sam.npy`, `*_compact_selection.npy`, `*_boosted.npy`,
`*_filled.npy`, `*_boosted_filled.npy`, `*_raw_masks.npz` (including IoU scores),
`*_summary.csv`, `*_notebook_results.json`, and `*_partitions.png`.
Selected original-mask indices and selection history are recorded in JSON.
Add `--create-gif` and/or `--save-step-images` for selection animation and frames.
These can consume substantial memory/storage; start with a single image.
This exploratory command overwrites files with the same image stem in its output folder.

## Reprocess saved raw masks without SAM

```bash
python src/scripts/run_notebook_pipeline.py \
  --image data/sample_images/truck.jpg \
  --raw-masks results/notebook_pipeline/truck/truck_raw_masks.npz \
  --select-count 100 \
  --output-dir results/notebook_pipeline/truck_select100
```

Input NPZ needs `masks` of shape `[N,H,W]` and `predicted_iou` of shape `[N]`.
Older single-image NPZ files contain no scores. For those, explicitly add
`--uniform-iou`; selection then uses new area only, not the notebook's full score.
Never substitute a 2D label map for a raw overlapping mask stack.
COCO/ImageNet batch outputs currently retain label maps, not raw mask stacks;
view them directly, or rerun one image to obtain scored raw masks.

## Notebook workflow versus current batch generation

The existing COCO/ImageNet batch scripts use connected-component refinement and
a region budget. The notebook pipeline uses greedy selection and SegmentBooster,
leaves zero as unassigned, and does not cap the final number of labels. They are
different experiments. These scripts do not silently change ongoing batch runs.
Generation/viewing remains separate from [XAI](../XAI/README.md).

The notebook loads all images twice before slicing to 300. The scripts instead
load one image at a time. Kaggle installation, archive, and display cells are
not executable script steps. A misspelled `compact_selction` filename is corrected
to `compact_selection`; saved-mask viewing uses the actual boosted filenames.
GIF frames use fixed dimensions to avoid the original variable-size frame error.

## Function inventory

Cell numbers below are zero-based notebook cell indices.

| Function                                  | Cell | Role                                                                                   |
| ----------------------------------------- | ---- | -------------------------------------------------------------------------------------- |
| `show_all_masks`                          | 9    | Original image, label map, and raw-mask overlay.                                       |
| `show_compact_overlay`                    | 11   | Label-map and image overlay.                                                           |
| `black_rainbow_colormap`                  | 13   | Discrete label colors; notebook treats zero as background.                             |
| `show_saved_masks`                        | 14   | Notebook-stage viewer; use the new viewer CLI for headless export.                     |
| `create_compact_mask`                     | 21   | Large masks first, small masks overwrite overlapping pixels.                           |
| `compact_mask_to_masks`                   | 23   | Convert nonzero labels back into binary-mask dictionaries.                             |
| `refiner`                                 | 25   | SegmentBooster: assign each mask its most frequent compact label.                      |
| `expand_matrix`                           | 27   | Legacy local expansion; retained for reference, not used by the main pipeline.         |
| `partition_unassigned_by_nearest_cluster` | 28   | Fill zero pixels by nearest label, optionally limited by distance/ties.                |
| `select_masks_by_unpicked_area`           | 31   | Greedy new-area × predicted-IoU selection, optional GIF and step images.               |
| `compute_coverage`                        | 33   | Count nonzero pixels and their percentage.                                             |
| `save_results`                            | 35   | Save notebook stage arrays and JSON; portable paths and corrected filename typo.       |
| `generate_at_least_n_masks`               | 37   | Combined notebook threshold search and downstream stages; the new CLI separates these. |

`expand_matrix` is preserved as written: its neighborhood condition excludes
same-row/same-column neighbors. It is a legacy alternative, not the distance-fill
implementation. The retained combined notebook function also hardcodes its
selection count to 400; the CLI exposes that separately as `--select-count`.

## Local image tests

Run the reproducible three-photo postprocessing demo:

```bash
python src/scripts/demo_notebook_pipeline.py
python -m unittest discover -s tests -p 'test_*pipeline.py'
```

Test photos are bundled scikit-image samples (astronaut, coffee, chelsea), resized
to 128 × 96. Masks are deterministic image-threshold fixtures, **not SAM predictions**.
The run checks postprocessing and produces artifacts under
`results/notebook_pipeline_demo/`. It does not validate SAM quality or GPU inference.

| Photo     | Selected fixture masks | Raw nonzero coverage | Boosted + filled nonzero coverage |
| --------- | ---------------------- | -------------------- | --------------------------------- |
| astronaut | 2                      | 80.54%               | 91.78%                            |
| coffee    | 2                      | 95.36%               | 97.66%                            |
| chelsea   | 2                      | 98.93%               | 100.00%                           |

The tests also cover greedy ordering, duplicate masks, overlap precedence,
mask/label conversion, bounded filling, ties, empty masks, GIF/frame export,
and rejecting mismatched image/map dimensions.

Dependencies: numpy, scipy, Pillow, matplotlib, scikit-image, and imageio.
Only live inference needs PyTorch/SAM2. Install `imageio` if it is missing.
