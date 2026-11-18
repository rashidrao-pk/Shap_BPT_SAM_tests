## Generate Partitions for `MS-COCO` Dataset

- [Connect first](../EPITO.md#1-start-a-gpu-session)
- [Download the dataset](../../DATASETS/MS_COCO.md)
- Download checkpoints: [SAM-1](../../MODELS/SAM1.md), [SAM-2](../../MODELS/SAM2.md)

### Check paths

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito --limit 2 --dry-run
```

### SAM-1

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
  --limit 1000
```

### SAM-2

#### Large Model

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
  --limit 1000
```

Remove `--limit 1000` for all 5,000 val2017 images. Repeat the same command to resume: completed images are skipped.

### Specific images

Add `--image-id ID [ID ...]` to generate only those images, for example to redo or inspect one:

```bash
python src/scripts/generate_coco_partitions.py --config MSCOCO_epito_sam1 --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" --sam1-variant vit_b --device cuda \
  --generate_at_least_n_masks 200 --save-partitions sam compact --image-id 76211 28285 57725 113867 11615
```

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda --generate_at_least_n_masks 200 --save-partitions sam compact --image-id 76211 28285 57725 113867 11615
```

COCO IDs work padded or not (`76211` = `000000076211`); other datasets use the file stem, e.g. `--image-id "benign (12)"`. Unknown IDs stop the run before SAM loads. The output folder and resume settings are the same as for a full run, so the images join the normal folder, and a later full run skips them. Completed images are skipped unless `--overwrite` is given. Every `generate_*_partitions.py` has this flag.

### Choose the partitions for XAI

| Model       | Folder                                  |
| ----------- | --------------------------------------- |
| SAM-1       | `MSCOCO_sam1_vit_b_254_compact-sam_200_greedy` |
| SAM-2 Large | `MSCOCO_sam2_large_254_compact-sam_200_greedy` |

`data.mask_dir_final` selects the folder used by XAI. `MSCOCO_epito` currently selects SAM-1.

**Folder suffix (since 2026-11-18).** When the `compact` stage is saved, the folder name ends with the compact strategy: `_greedy` (the default) or `_mso_<α>`. Every config points to the `_greedy` folders.
- **Same algorithm as before:** greedy is the selection the compact stage always used. Old 254-region folders without the suffix therefore hold greedy partitions, and renaming them keeps them.
- **Resuming in a renamed folder:** it works, because stamps written before the flag existed count as greedy.
- **The 62-region folders:** they keep their old names.

Rename once on the login node. A folder that already has a `_greedy` twin, for example from a test run, is merged into it without overwriting files:

```bash
cd /beegfs/home/mrashid/datasets/SAM_BPT
for d in *_254_*compact*; do
  [[ -d $d && $d != *_greedy && $d != *_mso_* && $d != *.log ]] || continue
  if [[ -d ${d}_greedy ]]; then
    find "$d" -maxdepth 1 -type f -exec mv -n -t "${d}_greedy" {} +
    rmdir "$d" 2>/dev/null && echo "$d merged into ${d}_greedy" || echo "kept $d: files also in ${d}_greedy (see mv messages)"
  else
    mv "$d" "${d}_greedy" && echo "$d -> ${d}_greedy"
  fi
done
```

Run it first with `echo` in front of `mv` and `find` to see what it would do. If it reports `kept`, some image appears in both folders. The `_greedy` copy is kept, and the files listed by `mv: not replacing` stay in the old folder. When both used the same settings (same folder name apart from the suffix), the old folder can be removed:

```bash
ls MSCOCO_sam2_large_254_compact-sam_200      # only the clashing files should remain
rm -r MSCOCO_sam2_large_254_compact-sam_200
```

## Preview the first five completed images

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --limit 5 \
  --methods sam compact
```

- Or preview specific images by COCO ID (one or more, padded or not). An image without partitions is reported and skipped, and the others still render:

```bash
python src/scripts/view_partitions.py \
  --config MSCOCO_epito \
  --sam sam2 \
  --image-id 76211 28285 57725 113867 11615 \
  --methods sam compact
```

- Choose the model with `--sam sam1` or `--sam sam2`. These read `data.mask_dir_final_sam1` / `_sam2` from the config. Without `--sam`, the preview uses `data.mask_dir_final` (SAM-1 for `MSCOCO_epito`). Use `--partition-folder NAME` for any other folder, for example an older `MSCOCO_sam1_vit_b_62_compact-sam_200`:

```bash
python src/scripts/view_partitions.py --config MSCOCO_epito --image-id 76211 --methods sam compact --sam sam2
```

The lower-left panel shows every ground-truth object of the image: one colour per object on black, with its class name, and the total count in the title. Objects are painted largest first, so small ones stay visible. Datasets without ground truth show the original there instead. `--no-gt-panel` restores the old layout.

Previews are saved under `previews/<partition folder>/`, so SAM-1 and SAM-2 previews of the same image do not overwrite each other.

## Compare compact strategies (greedy vs MSO)

`--compact-strategy` chooses how the compact stage picks SAM masks. Each step takes the mask with the highest score:

| Strategy | Score | Folder suffix |
| --- | --- | --- |
| `greedy` (default) | uncovered pixels × predicted IoU | `_greedy` |
| `mso` | uncovered pixels<sup>α</sup> × predicted IoU, with `--mso-alpha` α in (0, 1), default 0.5 | `_mso_<α>` |
| `split` | split gain × predicted IoU (see below); stops when no mask gains `--min-region-pixels` (default 64) | `_split` (`_split_<N>` for another minimum) |

With α < 1, MSO favours smaller masks more than greedy does, so it usually picks more, smaller regions. Each setting writes to its own folder, so the runs do not overwrite each other.

**Why greedy and MSO can end with very few regions.** Both count only *uncovered* pixels and stop once every pixel is covered. SAM often returns one mask for a whole facade, wall or table. Once that mask is picked, every window, door or object inside it adds no uncovered pixel. Its score becomes 0 and selection stops. An image with 212 SAM masks can then end with 13 compact regions, and MSO cannot change this, because it only reorders the picks.

**`split` fixes this.** The partition is painted as before, smaller masks on top of larger ones. The **split gain** of a mask counts:
- every unassigned pixel it covers;
- for every region it cuts, the smaller of the two parts (pixels it takes vs pixels left).

A window inside the facade therefore gains its own area. A mask that only repeats an existing region gains nothing and is skipped. Selection stops at `--max-regions` or when no mask gains at least `--min-region-pixels`. Expect many more regions than greedy, but fewer than the raw `sam` map, which keeps every mask.

**1. Generate on a GPU node** (after `source src/hpc/node_env.sh`). The same images go through each setting. SAM runs again for each one, so keep the image list short:

```bash
IDS="76211 28285 57725 113867 11615"
COMMON=(--config MSCOCO_epito --model sam2 --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt"
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml --device cuda
        --generate_at_least_n_masks 200 --save-partitions sam compact --image-id $IDS)
python src/scripts/generate_coco_partitions.py "${COMMON[@]}" --compact-strategy greedy
python src/scripts/generate_coco_partitions.py "${COMMON[@]}" --compact-strategy mso --mso-alpha 0.5
python src/scripts/generate_coco_partitions.py "${COMMON[@]}" --compact-strategy split
# Optional, to see the effect of alpha:
python src/scripts/generate_coco_partitions.py "${COMMON[@]}" --compact-strategy mso --mso-alpha 0.3
python src/scripts/generate_coco_partitions.py "${COMMON[@]}" --compact-strategy mso --mso-alpha 0.7
```

Each run prints its folder (`For XAI set data.mask_dir_final: "..."`), e.g. `MSCOCO_sam2_large_254_compact-sam_200_greedy` and `MSCOCO_sam2_large_254_compact-sam_200_mso_0.5`.

**2. Preview each folder** (no GPU needed):

```bash
F=MSCOCO_sam2_large_254_compact-sam_200
for S in greedy mso_0.5 split; do
  python src/scripts/view_partitions.py --config MSCOCO_epito --partition-folder ${F}_$S \
    --image-id $IDS --methods sam compact
done
```

**3. Compare them side by side** (no GPU needed):

```bash
python src/scripts/compare_partition_folders.py --config MSCOCO_epito --name compact_strategy \
  --folders ${F}_greedy ${F}_mso_0.5 ${F}_split --image-id $IDS
```

The comparison writes the following to `<output.evaluations>/partition_compare/compact_strategy/`:

- **One figure per image (`<id>.png`):**
  - Top row: the original, the raw SAM map, and each setting's compact map. Each map is titled with `SAM → Compact` region counts and the mean GT IoU next to a same-size grid's; ★ marks the best setting.
  - Bottom row: the GT objects, then overlays with the share of unassigned (black) pixels and of objects reaching IoU ≥ 0.5.
- **`per_image.csv`:** one row per image and setting.
- **`summary.csv`:** the same columns over all images, plus `best_on_images`, the number of images where the setting had the highest mean IoU. It is also printed as a table.

The IoU is the best union of cells for each GT object, as in [EVALUATE_PARTITIONS.md](../EVALUATE_PARTITIONS.md). More regions raise it almost automatically, so read it with the region count and the `gain` over the grid. `--labels` renames the columns; without `--image-id` it compares every image present in all folders (`--limit N`).

## Evaluate quality and weak greedy cases

The plain command below evaluates `data.mask_dir_final`, currently **SAM-1 greedy**, even when the latest generation run used SAM-2 split:

```bash
python src/scripts/evaluate_partitions.py --config MSCOCO_epito
```

To compare SAM-2 strategies using the new folders, select them explicitly. No GPU or model checkpoint is needed:

```bash
F=MSCOCO_sam2_large_254_compact-sam_200
python src/scripts/evaluate_partitions.py --config MSCOCO_epito \
  --folders ${F}_greedy ${F}_mso_0.5 ${F}_split \
  --baseline-folder ${F}_greedy \
  --output-dir results/evaluations/partition_quality/MSCOCO_sam2_strategies
```

Only list folders you have generated: remove `${F}_mso_0.5` to compare greedy with split alone. Folder names must match the generation output, including the region cap (`254` above).

With `--baseline-folder`, the evaluator selects images present in **every listed folder before applying `--limit`**. You can run it while split generation continues; the report covers only the shared images available then. Repeat it when generation finishes for the full comparison. Missing or unreadable images are reported, and fewer than 30 scored images give no overall USE/CAUTION/SKIP verdict.

Outputs include:

- `partition_quality_summary.csv` and `conclusion.md`: GT oracle IoU, coverage, region counts and the grid comparison per folder.
- `strategy_comparison.csv` and `strategy_comparison.md`: paired IoU changes against greedy, both overall and on weak greedy images. An image is weak when **any GT object has greedy IoU below 0.5**; change this with `--weak-iou`. The report lists the worst weak image IDs.
- `diagnosis.md`, `examples/` and `worst/`: failure groups and representative previews.

To draw the same weak images side by side, copy their IDs from `strategy_comparison.md`:

```bash
python src/scripts/compare_partition_folders.py --config MSCOCO_epito \
  --folders ${F}_greedy ${F}_mso_0.5 ${F}_split \
  --image-id 76211 28285 57725 113867 11615 --name weak_greedy
```

The IDs above are the earlier inspection examples; replace them with the measured weak IDs. Higher oracle IoU means the cells can represent GT better. Read it alongside region counts and grid gain: split may improve IoU partly by retaining more cells. A comparison restricted to weak greedy cases describes those cases, not the entire dataset.

Use `--all-sam` instead of `--folders` for a separate quality summary of every discovered MS-COCO SAM folder, including older region caps and strategies. Without `--baseline-folder`, each folder uses its own completed images.

## Progress

```bash
python src/scripts/check_partition_progress.py --config MSCOCO_epito --limit 1000
```

Next: [MS-COCO XAI](../../XAI/DATASETS/MSCOCO.md).
