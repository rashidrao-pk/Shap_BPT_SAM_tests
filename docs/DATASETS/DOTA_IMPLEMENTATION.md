# DOTA (remote sensing, oriented-box detection): implementation spec

**Audience:** a coding agent working in `Shap_BPT_SAM_tests` (repo root = working directory).
**Goal:** add DOTA-v1.0 as a new dataset with the same lifecycle as the other added datasets:
download dataset → download checkpoint → generate SAM partitions → evaluate partitions against GT → run XAI → ROAD and leaderboard → tests and docs.

Read this whole file first. Do the phases in order and stop at each **GATE** until it passes. Everything marked **VERIFY** is something the author could not check (no network or weights where this was written). Check it yourself; do not assume it.

---

## 0. Design decisions (already made, do not re-litigate)

| Topic | Decision | Why |
| --- | --- | --- |
| Image unit | **1024×1024 tiles** cut from DOTA val images. One tile = one "image" everywhere (partitions, XAI, GT). | Source images are up to thousands of pixels wide. Full-size SAM and 1000-evaluation explanations would be too slow. ISIC solves the same problem by pre-downscaling in `download_datasets.py`; this follows that pattern. 1024 is SAM's native input size and a standard DOTA detection size. |
| Split | **val** only (labels public). `test` labels are withheld, so `--split test` must fail with a clear message. | Need GT. |
| Model | Ultralytics **YOLO11s-OBB** (`yolo11s-obb.pt`), matching the repo's YOLO11s on COCO. | Pretrained on DOTA-v1.0, oriented boxes. |
| Runner | New runner **`OD_DOTA`** built on the shared `experiment_common.py` flow, modelled on **`OD_DETR`**. **Do not extend `OD_yolo.py`.** | `OD_yolo.py` is COCO-hardwired (numeric image ids, `.jpg`, COCO annotations, `result.boxes`) and has its own output layout, which ROAD and the budget studies do not support. The `OD_DETR` path gets ROAD, leaderboard, budget studies and `--explain_all` for free. |
| Detector score | Same as COCO detectors: for a fixed class, **sum of confidences of all detections of that class** (`--class-aggregation sum`, optional `max`). The explained class is the top class on the unmasked tile, fixed per tile. | Keeps results comparable with `OD_DETR` / `OD_yolo`. State in docs that a map explains the class on the tile, not one object. |
| GT | Polygon masks rasterised from the oriented boxes. Explanation GT = union of all instances of the explained class. Partition-evaluation GT = one mask per instance. | Same semantics as the COCO branch. |
| XAI methods | `BPT`, `BPT+SAM_c`, `SAM_c+KernelSHAP`, and with `--explain_all` also LIME, Shap-AA. **Gradient methods (LRP, GradCAM, IDG, GradExp, GradShap) raise `UnsupportedMethod` for detectors in `all_methods.py`; keep that behaviour.** | Existing repo rule. |
| Partition settings | SAM2 tiny, `--max-regions 62`, `--generate_at_least_n_masks 200`, `--save-partitions sam compact`. | 62 + background label 0 = the 63-label cap of the 64-bit signature. 200 matches COCO; aerial tiles hold many small objects. Folder: `DOTA_sam2_tiny_62_compact-sam_200`. |

Do **not**: change any existing dataset's behaviour or paths, change default studies, use GT while generating partitions, tune evaluation thresholds to get a better verdict, commit data or weights, or invent results. Report whatever the verdicts and metrics turn out to be.

---

## 1. Recon (read before writing code)

The repo patterns to mimic. Read them, then mirror every hit of `ISIC2018` (data side) and `OD_DETR` (detector side):

```bash
grep -rn "ISIC2018\|isic2018" src tests docs --include=*.py --include=*.yaml --include=*.md
grep -rn "OD_DETR" src tests --include=*.py
```

Known touch points (from a grep at commit `6a36122`). Treat as a checklist; the grep above is the source of truth.

| # | File | What to add |
| --- | --- | --- |
| 1 | `src/scripts/download_datasets.py` | `download_dota`, tiling function, `DATASETS['dota']`, `MODELS['dota']` |
| 2 | `src/configs/DOTA_epito.yaml` | New config |
| 3 | `src/scripts/generate_dota_partitions.py` | Copy of `generate_isic2018_partitions.py` with `default_config='DOTA_epito'` |
| 4 | `src/scripts/ground_truth.py` | `dota` format in `load_ground_truth` plus a per-instance helper |
| 5 | `src/scripts/evaluate_partitions.py` | `dota` branch in `gt_objects` (per-instance masks) |
| 6 | `src/scripts/explanations/model_adapters.py` | `YoloObbAdapter`, branch in `load_adapter` |
| 7 | `src/scripts/explanations/experiment_common.py` | `DEFAULT_CONFIGS`, `MODEL_NAMES`, `EXPECTED_DATASETS`, `CHECKPOINT_HELP`, and every `kind != 'OD_DETR'` / `kind == 'OD_DETR'` special case (see §7) |
| 8 | `src/scripts/explanations/OD_DOTA.py` | 10-line wrapper like `OD_DETR.py` |
| 9 | `src/scripts/summary_setup.py` | Model entry, `DATASETS['dota']`, GT-id branch for `format_name == 'dota'` |
| 10 | `src/scripts/check_partition_progress.py` | Add `DOTA_epito` to the default config list |
| 11 | `src/hpc/prepare_full_budget_study.py` | `MODELS['DOTA']`; keep it out of `DEFAULT_DATASETS` |
| 12 | `src/scripts/explanations/budget_ablation.py`, `budget_steps.py`, `compare_auc_points.py` | Runner lists / the `'OD_DETR' if runner == 'OD_yolo'` mappings: make `OD_DOTA` behave like `OD_DETR` |
| 13 | `src/scripts/explanations/summarize_ablation_study.py` | Display name `'OD_DOTA': 'DOTA'` |
| 14 | `src/scripts/explanations/xai_leaderboard.py` | Domain map: `'DOTA': 'remote sensing'` (new domain; check the plotting code copes with one more) |
| 15 | `src/scripts/explanations/evaluate_road.py`, `road.py` | Check how runners are registered; add `OD_DOTA` wherever `OD_DETR` is accepted |
| 16 | `docs/` | See §10 |
| 17 | `tests/test_dota.py` | See §9 |

Also read: `docs/DATASETS/ISIC_2018.md`, `docs/Partitions/DATASETS/ISIC2018.md`, `docs/XAI/DATASETS/MSCOCO.md`, `docs/Partitions/EVALUATE_PARTITIONS.md`, `docs/XAI/ROAD.md`, `docs/XAI/LEADERBOARD.md`, `docs/OUTPUT_PATHS.md`, `tests/test_added_datasets.py`.

---

## 2. Environment

```bash
python -c "import ultralytics, cv2, torch; print(ultralytics.__version__)"   # install ultralytics if missing
python src/scripts/summary_setup.py                                          # baseline, on a GPU node
pytest -q tests/test_added_datasets.py tests/test_od_yolo.py tests/test_ground_truth.py tests/test_evaluate_partitions.py   # must pass before AND after your changes
```

Add `ultralytics` to the requirements file that already lists it for `OD_yolo` (find it; do not create a new one). `cv2` is already in `PARTITION_PACKAGES`.

---

## 3. Phase 1: download the dataset and build tiles

### 3.1 Raw data

**Source A (scripted, preferred):** the Ultralytics `DOTAv1.zip` referenced by Ultralytics' `DOTAv1.yaml`.
**VERIFY:** the exact URL (believed to be `https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip`), its size, its folder layout (`images/{train,val,test}`, `labels/{train,val,test}`), and the label format (believed: one object per line, `class_index x1 y1 x2 y2 x3 y3 x4 y4`, coordinates **normalised to 0–1**, class index per the YAML `names`). Look at `ultralytics/cfg/datasets/DOTAv1.yaml` in the installed package for the authoritative URL and class names.
Source A labels carry **no `difficult` flag**. Say so in the docs.

**Source B (manual fallback):** the official DOTA release (`labelTxt` with pixel coordinates and a difficulty column). Do not script this unless you can confirm a stable, license-compatible direct URL; otherwise document manual download and the expected folder layout, and let the tiler accept it through `--label-format dota-txt`. Keep Source A as the tested path.

Reuse `download()`, `check_hash()` and `extract()` from `download_partimagenet.py`, exactly as `download_paco` does. After the first successful download, **compute the SHA-256 and pin it** in the script (as the other datasets do). Do not leave `None`.

Licence/terms: DOTA imagery comes from several sources (including Google Earth). **VERIFY** the current terms on the DOTA site and put the actual restriction (believed: academic use only) in `docs/DATASETS/DOTA.md`. Do not assert a licence you did not read.

Target layout (root is `DATASET_ROOT`, i.e. `/beegfs/home/mrashid/datasets` by default):

```text
DOTA/raw/...                              # downloaded archive and extraction
DOTA/prepared_1024/val/images/<tile>.png  # selected tiles
DOTA/prepared_1024/val/labels/<tile>.txt  # tile-local GT
DOTA/prepared_1024/val/tile_index.csv     # provenance + selection record
```

### 3.2 Tiler (`prepare_dota`, mirrors `prepare_isic`)

CLI additions to `download_datasets.py` (existing flags unchanged; ignore these for other datasets):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tile` | 1024 | Tile side |
| `--stride` | 1024 | Window step. Default **no overlap**, so one object is not explained twice. The standard training split uses overlap; do not use it here. |
| `--min-keep` | 0.7 | Minimum fraction of an object's area that must lie inside the tile for it to count as GT |
| `--min-objects` | 3 | Minimum number of counted objects for a tile to be a candidate |
| `--max-tiles` | 300 | Number of tiles to keep |
| `--seed` | 0 | Selection seed |

Algorithm:

1. For each val image, slide windows with `x0 = 0, stride, 2*stride, …`; clamp the last window inward (`x0 = max(0, W - tile)`) so every tile is full-size. If a side is smaller than `tile`, pad right/bottom with black; padding has no GT.
2. Tile id: `<source-stem>__<x0>_<y0>` (e.g. `P0003__2048_1024`). IDs must be filesystem- and `str.split("_", 1)`-safe for the rest of the pipeline: check `check_partition_progress.py` and `generate_coco_partitions.py` for how `id` is derived from the file stem in `input_format: directory`, and for any place that assumes digits or a single underscore.
3. For each object polygon (convert normalised coordinates to pixels with the **source** image size; polygon is 4 points), clip it against the tile rectangle with a small NumPy Sutherland–Hodgman routine (the box is convex, so this is exact; avoid adding a geometry dependency), and compute areas with the shoelace formula. Translate to tile coordinates.
   - kept area / original area ≥ `--min-keep` → counted object, `difficult=0`.
   - 0 < ratio < `--min-keep` → written with `difficult=1`. These are **ignored regions** (excluded from GT masks and metrics), so a half-cut plane the detector can still see does not count as a false localisation.
   - ratio = 0 → dropped.
4. Candidate tiles: ≥ `--min-objects` counted objects. Sort ids, then `random.Random(seed).sample` up to `--max-tiles`. Write **all** candidates to `tile_index.csv` with a `selected` column (columns: `tile_id, source_image, x0, y0, width, height, n_counted, n_ignored, selected`); write only selected tiles to `images/` and `labels/`.
5. Label file, one object per line, tile pixel coordinates, class names with spaces (Ultralytics naming):
   `x1 y1 x2 y2 x3 y3 x4 y4 <class name with spaces> <difficult 0|1>`
   Parse from the right (last token = difficulty, preceding tokens after the 8 numbers = class name) because class names contain spaces. Use a `\t` or `;` separator instead if you prefer, but document it and test it.
6. Save images as lossless PNG. Idempotent: skip tiles that already exist.
7. Print: val images found, candidate tiles, selected tiles, objects per class among selected tiles, and the leakage note (§3.3).

DOTA-v1.0 classes (names as in the YAML; **VERIFY** against the installed YAML): plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool.

### 3.3 Leakage note (put in docs)

The pretrained OBB weights were trained on DOTAv1 **train**; val is its standard evaluation split, so treat results as a benchmark, not unseen data (same wording as the COCO val2017 note). **VERIFY** in Ultralytics' docs that val was not in training. DOTA test labels are not public, so there is no better option.

### GATE 1

```bash
python src/scripts/download_datasets.py --dataset dota --assets dataset --split val --max-tiles 20
ls "$DATASET_ROOT/DOTA/prepared_1024/val/images" | head
```

Pass when: tiles are 1024×1024; every image has a label file; a spot check overlay (draw polygons on 3 tiles, save to the scratchpad, look at them) shows boxes sitting on the objects; `tile_index.csv` is written; re-running changes nothing.

---

## 4. Phase 2: download the checkpoint

`YOLO11s-OBB`:

```bash
mkdir -p src/checkpoints
curl --fail --location -o src/checkpoints/yolo11s-obb.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt
```

**VERIFY** the URL (HEAD request; also try the asset list of the Ultralytics release). Then add to `download_datasets.py`:

```python
'dota': [(f'{ULTRALYTICS}/yolo11s-obb.pt', 'src/checkpoints/yolo11s-obb.pt', '<full sha256 after first download>')],
```

so that `python src/scripts/download_datasets.py --dataset dota --assets model` works. Path convention follows `docs/MODELS/YOLO.md` (`src/checkpoints/`).

Load check (no GPU needed):

```python
from ultralytics import YOLO
m = YOLO('src/checkpoints/yolo11s-obb.pt')
assert m.task == 'obb' and len(m.names) == 15
```

Record `m.names` in the docs. Create `docs/MODELS/YOLO_OBB.md` (same table layout as `YOLO.md`; limits: score sums all detections of a class; oriented boxes; trained on DOTAv1 train).

---

## 5. Phase 3: generate partitions

### 5.1 Config `src/configs/DOTA_epito.yaml`

```yaml
# Tiles prepared by download_datasets.py --dataset dota.
device: cuda
data:
  name: DOTA
  input_format: directory
  image_dir: /beegfs/home/mrashid/datasets/DOTA/prepared_1024/val/images
  mask_dir_final: DOTA_sam2_tiny_62_compact-sam_200
  ground_truth:
    format: dota
    root: /beegfs/home/mrashid/datasets/DOTA/prepared_1024/val/labels
model:
  checkpoint: src/checkpoints/yolo11s-obb.pt
output:
  partitions: /beegfs/home/mrashid/datasets/SAM_BPT
  explanations: /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/results/explanations
  evaluations: /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/results/evaluations
evaluation:
  bg-type: "full"
```

Mirror the exact keys of `ISIC2018_epito.yaml`. Check how `config_paths.resolve_path` treats a relative `model.checkpoint` (ISIC uses `checkpoints/...` relative to the repo root; YOLO lives in `src/checkpoints`). Use whichever resolves correctly.

### 5.2 Wrapper

`src/scripts/generate_dota_partitions.py`: identical to `generate_isic2018_partitions.py`, with the docstring "Generate SAM partitions for the configured DOTA tile split." and `default_config='DOTA_epito'`. Confirm that `generate_coco_partitions.py` accepts `input_format: directory` ids with double underscores; fix only if needed, without changing other datasets.

### 5.3 Commands (these go into `docs/Partitions/DATASETS/DOTA.md`, same structure as the ISIC page)

```bash
# Check paths
python src/scripts/generate_dota_partitions.py --config DOTA_epito --limit 2 --dry-run

# SAM-2 tiny
python src/scripts/generate_dota_partitions.py \
  --config DOTA_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda --max-regions 62 --generate_at_least_n_masks 200 \
  --save-partitions sam compact --limit 2

# SAM-1 (optional, for the SAM1-vs-SAM2 comparison)
python src/scripts/generate_dota_partitions.py \
  --config DOTA_epito --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" --sam1-variant vit_b \
  --device cuda --max-regions 62 --generate_at_least_n_masks 200 \
  --save-partitions sam compact --limit 2
```

Keep `--save-partitions sam compact` (other stages change the folder name). Remove `--limit` for all selected tiles; the command resumes. Preview and progress:

```bash
python src/scripts/view_partitions.py --config DOTA_epito --limit 5 --methods sam compact --show-gt
python src/scripts/check_partition_progress.py --config DOTA_epito
```

`--show-gt` in `view_partitions.py` goes through `load_ground_truth`, so it needs the §6 loader. The folder name does not contain split/tile/stride, so changing any of those requires a different `output.partitions` root; say so in the docs.

SAM checkpoints already exist for the other datasets (`docs/MODELS/SAM1.md`, `SAM2.md`); do not re-download.

### GATE 2

2 tiles generated with no errors, `<id>_compact.npy` and `<id>_results.json` present, label values ≤ 62, preview looks sensible (look at it). Note the number of regions per tile; tiles with far more than 62 objects will merge some, which is expected and goes in the docs.

---

## 6. Phase 4: ground truth (`ground_truth.py`)

Add format `dota`:

- `_dota_instances(root, stem, shape)`: parse `root/<stem>.txt` (format from §3.2), return a list of `{class_name, polygon, difficult}`; verify `shape[:2] == (1024, 1024)` or at least that polygons fall inside the image, else raise with the file path. Rasterise each polygon on its own with `cv2.fillPoly` (or PIL) into a boolean mask.
- `_load_dota(settings, root, image_path, shape, gt_class)` returns the same structure as the COCO branch of `load_ground_truth` (use `_result`):
  - `gt_class` may be `None`/`'all'`/`'foreground'` (all counted instances) or a class name. Normalise names (`lower`, `-`/`_` → space). Unknown name → `ValueError('Unknown DOTA class …')`. Known class absent from the tile → empty mask, `status='no matching objects'`.
  - Build `labels` by painting instances in descending area order, so small objects stay visible on top of large ones (harbors contain ships).
  - `difficult=1` objects: do **not** include in `mask`; mark their pixels with the repo's ignored-label constant. Find it: `grep -n "IGNORED_LABEL" src/scripts/*.py`. `_result(..., ignored=[...])` already supports an ignored list, reuse it.
  - Extra keys as COCO does: `instance_count`, `display_count`, `count_label='instances'`, `semantic_labels=True`, `label_names`, `semantic_classes` (the 15 names), `scope`.
- Wire `format_name == 'dota'` into `load_ground_truth` after the existing branches.
- `validate_config` already checks `ground_truth.root` exists; nothing else needed.

`gt_mask_index._mask_index` indexes png/bmp/tif only. The DOTA labels are `.txt`, so do not route through it. If `summary_setup.py` or `check_partition_progress.py` call it for GT ids, add a `dota` branch that globs `*.txt` stems (see the existing `isic2018` branch near line 253 of `summary_setup.py`).

`save_ground_truth` for detectors is called from `experiment_common.py` (§7); the `{image_id}_gt.json` it writes must stay JSON-serialisable (the existing tests check this).

---

## 7. Phase 5: evaluate partitions

### 7.1 Code

`evaluate_partitions.py`:

- `image_paths(data)` already falls back to a recursive glob of image extensions; tile PNGs work. Check `data['name']` is not required to be in a list anywhere else (`grep -n "name" src/scripts/evaluate_partitions.py`).
- Add a `dota` branch to `gt_objects` **before** the generic `load_ground_truth` fallback, yielding `(class_name, instance_mask, valid)` for every counted instance (non-difficult), one instance at a time (as the COCO branch does), so overlapping instances don't erase one another. `valid` = `None`, or the not-ignored mask if the existing code expects it. Match the generator's signature exactly.
- Register `DOTA_epito` wherever datasets are enumerated for `--all-sam` (look at how `partition_folders(config, all_sam)` and the config list are built).

### 7.2 Run

```bash
python src/scripts/evaluate_partitions.py --config DOTA_epito            # folder the config selects
python src/scripts/evaluate_partitions.py --config DOTA_epito --all-sam  # every SAM folder found
```

Outputs under `results/evaluations/partition_quality/`: `conclusion.md` (verdict), `partition_quality_summary.csv`, `partition_quality_objects.csv`, `diagnosis.md`, `failure_causes`, `iou_by_object_size`, `examples/DOTA.png`, `worst/DOTA.png`.

### 7.3 What to expect and how to report it

Verdict rules are in `docs/Partitions/EVALUATE_PARTITIONS.md` (`USE` / `CAUTION` / `SKIP` / `NEED PARTITIONS` / `NO GT`). **Do not edit the thresholds.** Aerial tiles hold many objects that are only a few pixels wide (small vehicles, ships), and the size bins in `diagnosis.md` call anything under 0.1% of the image "tiny" (≈1,000 px on a 1024² tile, about 32×32). Expect a large share of objects in the `Merged` failure cause and a lower median IoU than COCO. That is a legitimate finding about hierarchical partitions on small objects, not a bug. Write it up per class (large-footprint classes such as tennis court, baseball diamond, swimming pool, storage tank, plane vs. small vehicle) using the existing per-class table, and state the verdict as computed.

If the verdict is `SKIP` or `CAUTION`, still run XAI (it is a negative-result datapoint) but add the verdict to the head of `docs/XAI/DATASETS/DOTA.md`.

### GATE 3

`conclusion.md` has a DOTA row with a verdict and reason; the numbers are non-trivially different from the grid baseline columns; `worst/DOTA.png` and `examples/DOTA.png` were looked at and the GT outlines align with the objects. If GT and partition are visibly misaligned, the bug is in tiling or rasterisation, not in SAM. Fix that before continuing.

---

## 8. Phase 6: the detector adapter and `OD_DOTA` runner

### 8.1 `YoloObbAdapter` in `model_adapters.py`

Interface to satisfy (copy from `DetrAdapter`): `class_names` (list), `predict_batch(images) -> ndarray[len, n_classes]`, `detections(image) -> list[dict]`, `name`, `batch_size`, `aggregate`. `images` are `uint8` HxWx3 arrays.

Behaviour:

- Resolve weights: `args.checkpoint` if given, else `args.model` looked up in `src/notebooks/checkpoints`, `src/checkpoints`, `checkpoints` (same search list as `resolve_yolo_source` in `OD_yolo.py`; **do not import `OD_yolo.py`**, it is a 2,000-line COCO script; reimplement the few lines).
- Fail fast if `model.task != 'obb'`.
- Predict with `model.predict(source=list_of_arrays, imgsz=1024, conf=args.conf, verbose=False)`. **VERIFY the channel order:** Ultralytics treats NumPy input as **BGR**. Check what `OD_yolo.py` does (`load_image`, `predict_yolo_batch`) and what `experiment_common.py` hands to adapters (RGB). Convert RGB→BGR explicitly in the adapter, and add a unit test that fails if the order is wrong (e.g. a fake model recording the array it receives).
- **VERIFY the result API.** For OBB models, `result.boxes` is `None`; detections live in `result.obb` (`.cls`, `.conf`, `.xyxyxyxy`, `.xywhr`). The COCO helper `yolo_result_to_scores` reads `result.boxes` and would crash here, which is why this adapter must not reuse it. Write `obb_scores(result, class_count, aggregate)` that sums (or maxes) `conf` per `cls`, tolerates empty results, and ignores `cls >= class_count`.
- `--conf` (new option on the `OD_DOTA` parser only; default 0.25, the Ultralytics default and the value `OD_yolo` uses unless you find otherwise). Expose `--class-aggregation sum|max` exactly as the DETR runner does (it already exists on the shared parser: check; reuse it).
- `detections(image)`: list of `{'class_id','class_name','confidence','polygon': [[x,y]×4]}` for confidence ≥ 0.3. Check how `experiment_common.py` (≈ line 452) and the plotting code read `box` from detections; if they require an axis-aligned `box`, also supply `box=[xmin,ymin,xmax,ymax]` computed from the polygon, and draw the polygon where the code draws boxes if easy.
- Deterministic inference: `model.eval()`, no augmentation, `half=False`.
- Batch size: reuse `args.model_batch_size`.

### 8.2 `experiment_common.py` special cases

Introduce `DETECTOR_RUNNERS = {'OD_DETR', 'OD_DOTA'}` and replace each of these:

- `kind != 'OD_DETR'` at the `--show-gt` validation (≈ line 136): detectors skip the generic `validate_config`? **Check why DETR is excluded** (COCO GT has no `ground_truth` block). DOTA *does* have one, so for `OD_DOTA` still call `validate_config`.
- `kind != 'OD_DETR' and args.model != MODEL_NAMES[kind]` (≈ line 238): `OD_DOTA` accepts other OBB weights, e.g. `--model yolo11m-obb`. Add it to the detector set.
- `gt_class = (args.gt_class or prediction['explained_class']) if kind == 'OD_DETR'` (≈ line 657): include `OD_DOTA`.
- `score_type` (≈ line 596): `'detection confidence'` for both.
- Add `'OD_DOTA'` to `DEFAULT_CONFIGS` (`DOTA_epito`), `MODEL_NAMES` (`yolo11s-obb`), `EXPECTED_DATASETS` (`dota`), `CHECKPOINT_HELP` (`download_datasets.py --dataset dota --assets model`).
- Tiles where the detector finds **nothing** (all-zero score vector at conf 0.25): the target class is undefined. Skip the tile with a recorded reason (`no detections above conf`) rather than explaining class 0; follow how the runner already records skipped images, and report the skipped count in the run summary. Do not lower `--conf` automatically.

Also handle `--target-class`: for detectors it is a class index; keep working.

### 8.3 Runner

`src/scripts/explanations/OD_DOTA.py`:

```python
#!/usr/bin/env python3
"""OD_DOTA: explain YOLO11-OBB detections on DOTA tiles with BPT, SAM_c and Kernel SHAP."""
from experiment_common import create_parser, run


def parse_args():
    return create_parser('OD_DOTA').parse_args()


if __name__ == '__main__':
    run('OD_DOTA', parse_args())
```

### 8.4 Commands (into `docs/XAI/DATASETS/DOTA.md`)

```bash
# Smoke test
python src/scripts/explanations/OD_DOTA.py \
  --config DOTA_epito --show-gt --limit 1 --budget 1000 --no-heatmaps

# All perturbation-based methods (needs requirements-xai-all.txt)
python src/scripts/explanations/OD_DOTA.py \
  --config DOTA_epito --show-gt --limit 20 --budget 1000 --no-heatmaps --explain_all
```

Outputs, per `docs/OUTPUT_PATHS.md` conventions:

```text
<output.explanations>/OD_DOTA/<partition-folder>/bg_full_B1000/<tile-id>/
<output.evaluations>/OD_DOTA/<partition-folder>/bg_full_B1000/
```

Repeating the command resumes. `--dry-run` must validate inputs without loading models.

### GATE 4

- `--dry-run` passes; `--limit 1` produces attribution maps, AUC table, GT overlay and `auc_results_all_images_current_run.csv` with insertion/deletion, pointing accuracy and best IoU filled in.
- Look at the output image: the GT outline matches the class that was explained; the map is not blank.
- With `--explain_all`, LIME and Shap-AA are present; LRP/GradCAM/IDG/GradExp/GradShap are reported as unsupported (not crashes) exactly as for `OD_DETR`.

---

## 9. Tests (`tests/test_dota.py`, no weights, no GPU, no network)

Follow `tests/test_added_datasets.py` (tmp_path fixtures, `sys.path` inserts, pytest). Minimum set:

1. **Tiler:** a synthetic 2500×1500 image with known polygons (one fully inside a tile, one straddling a boundary with ratio ≥ 0.7, one with ratio < 0.7, one outside). Assert tile count, full-size tiles including the clamped last window, counted vs `difficult=1` objects, coordinates translated to tile space, deterministic selection for a fixed seed, and idempotence.
2. **Label parsing:** class names with spaces (`storage tank`, `ground track field`) round-trip; malformed line raises with the file name.
3. **GT loader:** per-class mask, `all` mask, unknown class error, absent class gives empty mask with status, small object stays visible over a large one, `difficult` pixels are excluded from `mask`, result JSON-serialisable after dropping `mask`/`label_map`.
4. **`gt_objects` for DOTA:** one mask per counted instance; overlapping instances both returned in full.
5. **Adapter scoring:** fake result objects with `.obb.cls/.conf` and `boxes=None`: sum vs max, empty result, out-of-range class; BGR/RGB order check with a fake model; non-OBB model rejected.
6. **Registration:** `DOTA_epito.yaml` parses and has the required keys; `OD_DOTA` is in `DEFAULT_CONFIGS/MODEL_NAMES/EXPECTED_DATASETS`; `create_parser('OD_DOTA')` builds; `generate_dota_partitions.py` parser defaults to `DOTA_epito`; `download_datasets.py` exposes `dota` and rejects `--split test` for it; `--dry-run` of the runner on a tiny fixture config works.
7. **Regression:** the existing suites listed in §2 still pass unchanged.

Run `pytest -q tests/test_dota.py` and then the whole `tests/` folder.

---

## 10. Phase 7: ROAD, leaderboard, budget studies

ROAD runs automatically at the end of each explained tile (`road_*` columns in `auc_results_all_images_current_run.csv`); nothing extra is needed. To score after the fact:

```bash
python src/scripts/explanations/evaluate_road.py \
  --runner OD_DOTA --config DOTA_epito --budget 1000 --bg-type full --limit 20
```

(Needs the §1 item 15 registration; `OD_DETR` is already supported there, so mirror it.) Then the leaderboard for the run and across datasets:

```bash
python src/scripts/explanations/xai_leaderboard.py dataset \
  results/evaluations/OD_DOTA/DOTA_sam2_tiny_62_compact-sam_200/bg_full_B1000
python src/scripts/explanations/xai_leaderboard.py all --budget 1000 --bg-type full
```

Optional (budget study, off by default):

```bash
python src/hpc/prepare_full_budget_study.py --repo . --study <folder> --datasets DOTA
```

### GATE 5

`road_summary.csv` and `winner.md` exist for the DOTA run; the cross-dataset leaderboard still builds with DOTA included as a "remote sensing" domain; existing datasets' leaderboard rows are unchanged.

---

## 11. Phase 8: scale-up (only after GATE 5)

1. Partitions for all selected tiles (resume-safe), then re-run `evaluate_partitions.py` on the full set.
2. XAI over all tiles with `--budget 1000 --explain_all --no-heatmaps`. Log per-tile wall time; the Kernel-SHAP/LIME methods are the slow ones. If a node has a time limit, use the repo's sbatch pattern (`docs/XAI/EPITO_SBATCH`).
3. Report in the run notes: number of tiles, number skipped for "no detections", verdict from partition evaluation, leaderboard `winner.md`.
4. Statistics caveat to put in the docs: tiles cut from the same source image are not independent. Report the number of **source images** next to the number of tiles, and prefer paired tests across methods per tile with source-image clustering if you add significance tests.

---

## 12. Documentation to add or update

New pages, copying the structure and tone of the ISIC/COCO pages (short, command-first, a "Limits on what you can claim" section):

- `docs/DATASETS/DOTA.md`: table (images, GT, model, download size, licence), the download+tiling command, tile design, leakage note, limits.
- `docs/MODELS/YOLO_OBB.md`: as §4.
- `docs/Partitions/DATASETS/DOTA.md`: as §5.3, folder names, progress, evaluation command.
- `docs/XAI/DATASETS/DOTA.md`: as §8.4, how the explained class is chosen, ground truth, reading the results, outputs and resume.

Update the index rows: `docs/DATASETS/README.md` (table row, "Added datasets" checklist, `check_partition_progress` command), `docs/Partitions/DATASETS/README.md`, `docs/XAI/README.md`, `docs/XAI/GROUND_TRUTH.md`, `docs/MODELS/README.md`.

Required statements in the docs (be exact, not hedged away):

- A map explains the **class on the tile**, not one object (scores sum over detections).
- GT is the union of oriented-box polygons: a box is coarser than the object's true outline, so IoU and pointing metrics are upper-bounded by box shape.
- Objects cut by a tile border and under `--min-keep` are ignored regions.
- Many DOTA objects are tiny; partitions cannot isolate them; see the partition verdict.
- SAM was trained on natural images; aerial imagery is a domain shift. This is untested here and is exactly what the partition evaluation measures.
- Source A labels have no `difficult` flag.
- Research diagnostics only.

---

## 13. Acceptance checklist

- [ ] `download_datasets.py --dataset dota --assets all` downloads, pins hashes, and builds tiles; `--split test` is rejected with a clear message.
- [ ] `generate_dota_partitions.py --config DOTA_epito --dry-run` passes; 2-tile SAM2 run produces `DOTA_sam2_tiny_62_compact-sam_200`.
- [ ] `view_partitions.py --show-gt` renders GT outlines on tiles.
- [ ] `evaluate_partitions.py --config DOTA_epito` writes a verdict, with per-class and size-bin diagnosis.
- [ ] `OD_DOTA.py --limit 1` and `--explain_all --limit 20` run; gradient methods are reported as unsupported, not crashes; no-detection tiles are skipped with a reason.
- [ ] ROAD columns present; leaderboard builds; cross-dataset leaderboard includes DOTA.
- [ ] `tests/test_dota.py` passes; the full `tests/` suite passes; no existing config, path, default study or output folder changed.
- [ ] Docs added and indexed; every **VERIFY** below resolved and either confirmed or corrected in the docs.
- [ ] A short `docs/XAI/DATASETS/DOTA.md` "Results so far" section contains only numbers that were actually produced by these runs.

---

## 14. Open items to VERIFY (author could not check these)

| # | Item | Where it matters |
| --- | --- | --- |
| 1 | `DOTAv1.zip` URL, size, folder layout, label normalisation, class order | §3.1 |
| 2 | DOTA usage terms (believed academic-only) | docs |
| 3 | `yolo11s-obb.pt` download URL; SHA-256 to pin | §4 |
| 4 | OBB result API (`result.obb`, `boxes is None`) in the installed Ultralytics version | §8.1 |
| 5 | NumPy input is BGR for Ultralytics predict; what `experiment_common` passes | §8.1 |
| 6 | Pretrained OBB weights trained on DOTAv1 train only (val held out) | §3.3 |
| 7 | Default `conf` that `OD_yolo` uses; keep the two consistent | §8.1 |
| 8 | `IGNORED_LABEL` constant and how `_result(..., ignored=...)` expects it | §6 |
| 9 | How `generate_coco_partitions.py` derives ids from tile file stems (double underscores) | §3.2, §5.2 |
| 10 | Registration in `evaluate_road.py` / `road.py` for a new detector runner | §10 |
| 11 | `xai_leaderboard.py` domain map and plotting with a 4th domain | §10 |
| 12 | Whether `relative model.checkpoint` resolves from `src/checkpoints` via `config_paths` | §5.1 |

---

## 15. Verification results (2026-11-09)

| # | Result |
| --- | --- |
| 1 | Confirmed: `https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip`, 1,988,510,024 bytes, SHA-256 `59e84c52…11d58b816` (pinned). Layout `DOTAv1/images/{train,val,test}`, `DOTAv1/labels/{train,val}` (class index + 8 normalised coordinates), **and `DOTAv1/labels/{train,val}_original`**. val: 458 `.jpg` images. Class order matches the YAML. |
| 1b | **Correction:** Source A *does* carry the difficult flag, in `labels/val_original` (official labelTxt: pixels, hyphenated class names, `imagesource`/`gsd` header lines). The tiler reads it by default (`--label-format auto`). |
| 2 | DOTA site: academic use only, commercial use prohibited; Google Earth images also under Google Earth terms. |
| 3 | Confirmed URL; 19,934,534 bytes; SHA-256 `43fa6310…f6b1a45a` (pinned). Not loaded locally (pickle); its training args were read statically: task `obb`, data `runs/DOTAv1-ms.yaml`, val split `val`, Ultralytics 8.2.100. |
| 4 | Confirmed in Ultralytics 8.4.190: OBB `Results(obb=...)`, `boxes` is None. |
| 5 | Confirmed: NumPy input is BGR. The adapter flips RGB→BGR (tested). Note: `OD_yolo.py` passes RGB arrays to `predict`, so existing COCO YOLO runs used swapped channels (left unchanged, reported). |
| 6 | **Not confirmable.** `DOTAv1-ms.yaml` is a local file; Ultralytics' multi-scale split tiles both train and val, and their headline mAP is from the test server. Docs say val may have been used. |
| 7 | `OD_yolo` passes no `conf`, so the Ultralytics default 0.25 applies; `OD_DOTA --conf` defaults to 0.25. |
| 8 | No named constant: ImageNet-S uses 1000. DOTA uses `dota_data.IGNORED_LABEL = 65535`, recorded in the GT as `ignored_labels`; `best_iou` now reads it (ImageNet-S unchanged). |
| 9 | IDs are the file stem; `P0003__123_0` works throughout. |
| 10 | `evaluate_road.RUNNERS` is `DEFAULT_CONFIGS`, so `OD_DOTA` is registered automatically. |
| 11 | Domain map extended; plots iterate over whatever domains exist. |
| 12 | Relative `model.checkpoint` resolves from the repository root; `src/checkpoints/yolo11s-obb.pt` is used. |

Local gates: GATE 1 on 4 real val images (tiles, overlays, idempotence); GATE 2 SAM-2 tiny on 2 real tiles (max label 62); GATE 3 alignment checked in `worst/` and the GT preview. GATE 4 ran with a fake OBB model only; GATE 5 and the full-scale run are for EPITO.
