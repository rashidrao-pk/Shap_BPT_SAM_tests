# DOTA explanations

Partition verdict: **not yet computed.** Run `python src/scripts/evaluate_partitions.py --config DOTA_epito` on the full tile set and record the verdict here before reading the XAI results. If it is `CAUTION` or `SKIP`, the explanations are still worth running, as a negative-result data point.

Run from the repository root on a GPU node. First [download the dataset and cut tiles](../../DATASETS/DOTA.md), get the [YOLO11s-OBB](../../MODELS/YOLO_OBB.md) checkpoint, and [generate partitions](../../Partitions/DATASETS/DOTA.md). Make sure `data.mask_dir_final` names the generated folder.

## Run

```bash
# Smoke test
python src/scripts/explanations/OD_DOTA.py \
  --config DOTA_epito --show-gt --limit 1 --budget 1000 --no-heatmaps

# All perturbation-based methods (needs requirements-xai-all.txt)
python src/scripts/explanations/OD_DOTA.py \
  --config DOTA_epito --show-gt --limit 20 --budget 1000 --no-heatmaps --explain_all
```

The three partition methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` adds LIME and Shap-AA. LRP, GradCAM, IDG, GradExp and GradShap have no detector adapter and are reported as unsupported, as for DETR. Add `--dry-run` to check paths without loading the model.

| Option | Default | Meaning |
| --- | --- | --- |
| `--conf` | 0.25 | Detection confidence threshold |
| `--class-aggregation` | `sum` | `sum` or `max` of a class's box confidences |
| `--target-class` | top class | Explain this class index instead |
| `--model` | `yolo11s-obb` | Other OBB weights, by name (see [YOLO11s-OBB](../../MODELS/YOLO_OBB.md)) |

## How the explained class is chosen

The detector runs on the whole tile. Each class's score is the sum of its box confidences, and the class with the highest score is explained, fixed for that tile. **A map explains the class on the tile, not one object:** with eight small vehicles, it explains all eight together.

A tile with no box above `--conf` has no class to explain. It is skipped with the reason `no detections above conf 0.25`, written to `<tile>_skipped.json` and `skipped.json`, and counted at the end of the run. The threshold is never lowered automatically.

## Ground truth

`--show-gt` shows every counted object of the explained class, filled from its oriented box. Results include pointing accuracy and best IoU. Best IoU picks the best threshold using GT, so it is a diagnostic, not a test score. Ignored regions (border-cut or difficult objects of that class) are left out of both.

## Reading the results

- GT is the filled box, coarser than the object. A perfect map of a plane's outline still misses the box corners, so localisation scores are capped below 1.
- Many objects are tiny. Check the [partition verdict](../../Partitions/EVALUATE_PARTITIONS.md) and its per-class table: classes with a large footprint (tennis court, baseball diamond, swimming pool, storage tank, plane) are a fairer test than small vehicles.
- SAM was trained on natural photographs; aerial tiles are a domain shift.
- The detector may have seen DOTA val in training (see [DOTA](../../DATASETS/DOTA.md#limits-on-what-you-can-claim)).
- Tiles from one source image are not independent. Report the number of source images (from `tile_index.csv`) next to the number of tiles.
- Research diagnostics only.

## ROAD and leaderboard

ROAD is scored during the run (`road_*` columns). For runs made without it:

```bash
python src/scripts/explanations/evaluate_road.py \
  --runner OD_DOTA --config DOTA_epito --budget 1000 --bg-type full --limit 20
```

The run ends with its leaderboard (`winner.md`). Across datasets, DOTA is the **remote sensing** domain:

```bash
python src/scripts/explanations/xai_leaderboard.py all --budget 1000 --bg-type full
```

## Outputs and resume

```text
<output.explanations>/OD_DOTA/<partition-folder>/bg_full_B1000/<tile-id>/
<output.evaluations>/OD_DOTA/<partition-folder>/bg_full_B1000/
```

Repeat the same command to resume: completed tiles are skipped. Skipped tiles are checked again with one forward pass. `--no-heatmaps` skips individual method images, and keeps the combined plots and numeric results.

## Results so far

None yet. Only numbers produced by real runs belong here.
