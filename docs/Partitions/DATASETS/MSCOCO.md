## Generate Partitions for `MS-COCO` Dataset

- [Connect first](../EPITO.md#1-start-a-gpu-session)
- [Download the dataset](../../SETUP/DATASETS/MS_COCO.md)
- Download checkpoints: [SAM-1](../../SETUP/MODELS/SAM1.md), [SAM-2](../../SETUP/MODELS/SAM2.md)

### Check paths

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito --limit 2 --dry-run
```

### SAM-1

`--compact-strategy greedy split` saves both compact selections from one SAM run: `<id>_compact_greedy.npy` and `<id>_compact_split.npy` in the same folder. `data.compact_strategy` in the config picks the one used by XAI (`split` in every config); see [Compare compact strategies](#compare-compact-strategies-greedy-vs-mso).

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito_sam1 \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda \
  --max-regions 254 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact \
  --compact-strategy greedy split \
  --limit 1000
```

### SAM-2

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --max-regions 254 \
  --generate_at_least_n_masks 200 \
  --save-partitions sam compact \
  --compact-strategy greedy split \
  --limit 1000
```

Remove `--limit 1000` for all 5,000 val2017 images. Repeat the same command to resume: completed images are skipped.

### Specific images

Add `--image-id ID [ID ...]` to generate only those images, for example to redo or inspect one:

```bash
python src/scripts/generate_coco_partitions.py --config MSCOCO_epito_sam1 --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" --sam1-variant vit_b --device cuda \
  --generate_at_least_n_masks 200 --save-partitions sam compact --compact-strategy greedy split --image-id 76211 28285 57725 113867 11615
```

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda --generate_at_least_n_masks 200 --save-partitions sam compact --compact-strategy greedy split --image-id 76211 28285 57725 113867 11615
```

COCO IDs work padded or not (`76211` = `000000076211`); other datasets use the file stem, e.g. `--image-id "benign (12)"`. Unknown IDs stop the run before SAM loads. The output folder and resume settings are the same as for a full run, so the images join the normal folder, and a later full run skips them. Completed images are skipped unless `--overwrite` is given. Every `generate_*_partitions.py` has this flag.

### Choose the partitions for XAI

| Model       | Folder                                  |
| ----------- | --------------------------------------- |
| SAM-1       | `MSCOCO_sam1_vit_b_254_compact-sam_200` |
| SAM-2 Large | `MSCOCO_sam2_large_254_compact-sam_200` |

`data.mask_dir_final` selects the folder used by XAI. `MSCOCO_epito` currently selects SAM-1.

**Compact files (since 2026-11-18).** Each compact selection is a file of its own in the partition folder:

| File                       | Made with                                |
| -------------------------- | ---------------------------------------- |
| `<id>_sam.npy`             | all SAM masks (one per image)            |
| `<id>_compact_greedy.npy`  | `--compact-strategy greedy` (default)    |
| `<id>_compact_mso_0.5.npy` | `--compact-strategy mso --mso-alpha 0.5` |
| `<id>_compact_split.npy`   | `--compact-strategy split`               |

The folder name does not depend on the strategy. Behaviour:

- **One run, several strategies:** `--compact-strategy greedy split` runs SAM once per image and writes both files.
- **Adding a strategy later:** run the same command with the new strategy added. Images that already have the other files run SAM again and write only the missing file; existing files are not touched.
- **Which file is used:** `data.compact_strategy` in the config (`greedy` by default). It applies to XAI, `view_partitions --methods compact`, the audit and the comparison. XAI results for a non-greedy strategy go to `<folder>_<strategy>`, so they never overwrite the greedy results.
- **Older folders:** a single `<id>_compact.npy` is read as greedy, so the 62-region folders and other older folders work unchanged.

**Merge the folders written while the strategy was a folder suffix** (`..._greedy`, `..._mso_0.5`, `..._split`). Stop any generator writing to them first. Then, on the login node, preview the merge and apply it:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
source /beegfs/home/mrashid/pt_312_sam_x86/bin/activate; unset PYTHONPATH
python src/hpc/merge_compact_folders.py --root /beegfs/home/mrashid/datasets/SAM_BPT           # dry run
python src/hpc/merge_compact_folders.py --root /beegfs/home/mrashid/datasets/SAM_BPT --apply
```

The merge works per image:

- `<base>_<tag>/<id>_compact.npy` becomes `<base>/<id>_compact_<tag>.npy`.
- The metadata are merged.
- A duplicate SAM map is dropped. A different one, from a separate SAM run, is kept as `<id>_sam_<tag>.npy` and flagged as `sam_run_differs` in the metadata.
- Nothing is overwritten, and folders with unfinished `*.tmp` files are skipped.

## Preview the first five completed images

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --sam sam2 \
  --limit 5 \
  --methods sam compact_greedy compact_split
```

- Or preview specific images by COCO ID (one or more, padded or not). An image without partitions is reported and skipped, and the others still render:

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --sam sam2 \
  --image-id 76211 28285 57725 113867 11615 \
  --methods sam compact_greedy compact_split
```

- Choose the model with `--sam sam1` or `--sam sam2`. These read `data.mask_dir_final_sam1` / `_sam2` from the config. Without `--sam`, the preview uses `data.mask_dir_final` (SAM-1 for `MSCOCO_epito`). Use `--partition-folder NAME` for any other folder, for example an older `MSCOCO_sam1_vit_b_62_compact-sam_200`:

```bash
python src/scripts/view_partitions.py --config MSCOCO_epito --image-id 76211 --methods sam compact --sam sam2
```

The lower-left panel shows every ground-truth object of the image: one colour per object on black, with its class name, and the total count in the title. Objects are painted largest first, so small ones stay visible. Datasets without ground truth show the original there instead. `--no-gt-panel` restores the old layout.

Previews are saved under `previews/<partition folder>/`, so SAM-1 and SAM-2 previews of the same image do not overwrite each other.

## Compare compact strategies (greedy vs MSO)

`--compact-strategy` chooses how the compact stage picks SAM masks. Each step takes the mask with the highest score:

| Strategy           | Score                                                                                               | File                                                        |
| ------------------ | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `greedy` (default) | uncovered pixels × predicted IoU                                                                    | `<id>_compact_greedy.npy`                                   |
| `mso`              | uncovered pixels<sup>α</sup> × predicted IoU, with `--mso-alpha` α in (0, 1), default 0.5           | `<id>_compact_mso_<α>.npy`                                  |
| `split`            | split gain × predicted IoU (see below); stops when no mask gains `--min-region-pixels` (default 64) | `<id>_compact_split.npy` (`_split_<N>` for another minimum) |

With α < 1, MSO favours smaller masks more than greedy does, so it usually picks more, smaller regions. Each strategy is saved as its own file in the same folder (see [Choose the partitions for XAI](#choose-the-partitions-for-xai)).

**Why greedy and MSO can end with very few regions.** Both count only _uncovered_ pixels and stop once every pixel is covered. SAM often returns one mask for a whole facade, wall or table. Once that mask is picked, every window, door or object inside it adds no uncovered pixel. Its score becomes 0 and selection stops. An image with 212 SAM masks can then end with 13 compact regions, and MSO cannot change this, because it only reorders the picks.

**`split` fixes this.** The partition is painted as before, smaller masks on top of larger ones. The **split gain** of a mask counts:

- every unassigned pixel it covers;
- for every region it cuts, the smaller of the two parts (pixels it takes vs pixels left).

A window inside the facade therefore gains its own area. A mask that only repeats an existing region gains nothing and is skipped. Selection stops at `--max-regions` or when no mask gains at least `--min-region-pixels`. Expect many more regions than greedy, but fewer than the raw `sam` map, which keeps every mask.

**1. Generate on a GPU node** (after `source src/hpc/node_env.sh`). One run computes every strategy from the same SAM masks:

```bash
IDS="76211 28285 57725 113867 11615"
python src/scripts/generate_coco_partitions.py --config MSCOCO_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda --generate_at_least_n_masks 200 --save-partitions sam compact \
  --compact-strategy greedy mso split --mso-alpha 0.3 0.5 0.7 --image-id $IDS
```

This writes `<id>_compact_greedy.npy`, `<id>_compact_mso_0.3.npy`, `<id>_compact_mso_0.5.npy`, `<id>_compact_mso_0.7.npy` and `<id>_compact_split.npy` into `MSCOCO_sam2_large_254_compact-sam_200`. Drop `--image-id` for the full run.

**2. Preview them side by side** (no GPU needed):

```bash
python src/scripts/view_partitions.py --config MSCOCO_epito --sam sam2 --image-id $IDS \
  --methods sam compact_greedy compact_mso_0.5 compact_split
```

**3. Compare them, scored against the GT** (no GPU needed):

```bash
python src/scripts/compare_partition_folders.py --config MSCOCO_epito --name compact_strategy \
  --folders MSCOCO_sam2_large_254_compact-sam_200 \
  --partition-types compact_greedy compact_mso_0.5 compact_split --image-id $IDS
```

The comparison writes the following to `<output.evaluations>/partition_compare/compact_strategy/`:

- **One figure per image (`<id>.png`):**
  - Top row: the original, the raw SAM map, and each setting's compact map. Each map is titled with `SAM → Compact` region counts and the mean GT IoU next to a same-size grid's; ★ marks the best setting.
  - Bottom row: the GT objects, then overlays with the share of unassigned (black) pixels and of objects reaching IoU ≥ 0.5.
- **`per_image.csv`:** one row per image and setting.
- **`summary.csv`:** the same columns over all images, plus `best_on_images`, the number of images where the setting had the highest mean IoU. It is also printed as a table.

The IoU is the best union of cells for each GT object, as in [EVALUATE_PARTITIONS.md](../EVALUATE_PARTITIONS.md). More regions raise it almost automatically, so read it with the region count and the `gain` over the grid. `--labels` renames the columns. Without `--image-id`, it compares every image that has all the listed files (`--limit N`). Several `--folders` with one `--partition-types` value compare folders instead, e.g. SAM-1 against SAM-2.

## Evaluate quality and weak greedy cases

The plain command below evaluates `data.mask_dir_final`, currently **SAM-1 greedy**, even when the latest generation run used SAM-2 split:

```bash
python src/scripts/evaluate_partitions.py --config MSCOCO_epito
```

To compare SAM-2 strategies, which are files in one folder, list them. No GPU or model checkpoint is needed:

```bash
F=MSCOCO_sam2_large_254_compact-sam_200
python src/scripts/evaluate_partitions.py --config MSCOCO_epito --folders $F \
  --partition-types compact_greedy compact_mso_0.5 compact_split --baseline-type compact_greedy \
  --output-dir results/evaluations/partition_quality/MSCOCO_sam2_strategies
```

Only list strategies you have generated: drop `compact_mso_0.5` to compare greedy with split alone. To score one strategy, use `--partition-type compact_split`, or set `data.compact_strategy: split` in the config.

With a baseline (`--baseline-type`, or `--baseline-folder` for folders), the evaluator selects images present in **every compared file set before applying `--limit`**. You can run it while generation continues; the report covers only the shared images available then. Repeat it when generation finishes for the full comparison. Missing or unreadable images are reported, and fewer than 30 scored images give no overall USE/CAUTION/SKIP verdict.

Outputs include:

- `partition_quality_summary.csv` and `conclusion.md`: GT oracle IoU, coverage, region counts and the grid comparison per folder.
- `strategy_comparison.csv` and `strategy_comparison.md`: paired IoU changes against greedy, both overall and on weak greedy images. An image is weak when **any GT object has greedy IoU below 0.5**; change this with `--weak-iou`. The report lists the worst weak image IDs.
- `diagnosis.md`, `examples/` and `worst/`: failure groups and representative previews.

To draw the same weak images side by side, copy their IDs from `strategy_comparison.md`:

```bash
python src/scripts/compare_partition_folders.py --config MSCOCO_epito --folders $F \
  --partition-types compact_greedy compact_mso_0.5 compact_split \
  --image-id 76211 28285 57725 113867 11615 --name weak_greedy
```

The IDs above are the earlier inspection examples; replace them with the measured weak IDs. Higher oracle IoU means the cells can represent GT better. Read it alongside region counts and grid gain: split may improve IoU partly by retaining more cells. A comparison restricted to weak greedy cases describes those cases, not the entire dataset.

Use `--all-sam` instead of `--folders` for a separate quality summary of every discovered MS-COCO SAM folder, including older region caps. Each folder is scored with `data.compact_strategy`, or with `--partition-type`. Without a baseline, each folder uses its own completed images.

## Progress

```bash
python src/scripts/check_partition_progress.py --config MSCOCO_epito --limit 1000
```

Next: [MS-COCO XAI](../../XAI/DATASETS/MSCOCO.md).
