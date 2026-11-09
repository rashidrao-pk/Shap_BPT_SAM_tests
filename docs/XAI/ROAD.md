# ROAD evaluation

ROAD ([Rong et al., ICML 2022](https://arxiv.org/abs/2202.00449)) tests whether an attribution map ranks pixels by how much they matter, without the sharp holes that black or gray replacement leaves.

It removes a fixed share of pixels, fills them in from their neighbours, and re-scores the model. Filling in hides the shape of the removed area, so the score reacts to lost information rather than to a visible mask. This matters for partition-based methods, whose regions follow object edges.

## Run

### During the explanations (default)

Every explanation runner scores ROAD right after each image's maps are made, and writes the `road_*` columns into `auc_results_all_images_current_run.csv`. Nothing extra is needed:

```bash
python src/scripts/explanations/OC_CelebA.py \
  --config CelebAHQ_epito --attribute Brown_Hair \
  --show-gt --limit 1 --explain_all --no-heatmaps --no-auc --no-resume
```

This works with `--no-auc` too. `--no-road` turns it off, and the `--road-*` settings below work here as well. ROAD adds about 2–5 seconds per map on a CPU, so roughly 30 seconds per image with `--explain_all`.

ROAD options don't count as explanation settings. Turning ROAD on or off, or changing a `--road-*` value, never makes a finished run recompute its explanations. Images finished before ROAD existed, or with `--no-road`, resume without ROAD; the run prints how many, and the separate command below adds them.

### After the explanations

ROAD reads the saved `*_attributions.npz` files, so nothing is regenerated:

```bash
python src/scripts/explanations/evaluate_road.py \
  --runner OC_ResNet --config ImageNet_epito \
  --budget 1000 --bg-type full --show-gt --gt-only --limit 2
```

Use the same `--config`, `--budget`, `--bg-type`, and image selection as the explanation run. Keep the runner's own options too, such as `--checkpoint`:

```bash
python src/scripts/explanations/evaluate_road.py \
  --runner OC_PACO --config PACO_epito --budget 1000 --limit 2 \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth --weights IMAGENET1K_V2
```

Runners: every `OC_*` runner, `COD_ViT`, and `OD_DETR`. `OD_yolo` is not supported, because it has its own runner and output layout.

Repeat the command to resume. `--methods` limits the run to some saved maps. Changing a `--road-*` setting recomputes the results.

## Results

| File | Location | Content |
| --- | --- | --- |
| `<image-id>_road.json` | Next to the attributions | Scores at each removal level |
| `road_results.csv` | Run's evaluation folder | One row per image and method |
| `road_summary.csv` | Run's evaluation folder | Mean and standard deviation per method |
| `auc_results_all_images_current_run.csv` | Run and evaluation folders | The AUC table, with `road_morf`, `road_lerf`, `road_gap` and their `_rel` forms added |

The ROAD columns are matched to the AUC rows by `image_id` and `method`. A method without a saved map, such as one the run did not save, gets empty ROAD cells. A run with `--limit` or `--methods` keeps ROAD values scored earlier for the other rows.

When explanations resume, the ROAD scores saved in `<image-id>_road.json` go back into the rebuilt table. An image whose explanations are recomputed, for example with `--no-resume`, always gets fresh ROAD scores. The separate command does not check whether a map changed, so after regenerating maps with `--no-road`, delete the old `<image-id>_road.json` files first.

| Metric | Removes first | Better | Meaning |
| --- | --- | --- | --- |
| `morf` | Most relevant pixels | Lower | The score should fall quickly |
| `lerf` | Least relevant pixels | Higher | The score should hold |
| `gap` | | Higher | `lerf - morf`, the main comparison |

Each value is the mean target score from the unmodified image up to 90% removal. Columns ending in `_rel` divide by the unmodified score, so images with different confidence can be averaged. They are empty when that score is not positive, as with CelebA logits.

## How methods are kept comparable

- Every method removes exactly the same number of pixels at each level, 10% to 90%. Region-based maps do not get fewer points, unlike the insertion/deletion AUC.
- Equal attribution values are ordered by one random order per image, shared by all methods.
- Noise is shared too, so differences between methods do not come from the random seed.

## Settings

| Option | Default | Meaning |
| --- | --- | --- |
| `--road-fractions` | 0.1 to 0.9 | Removal levels |
| `--road-noise` | 0.01 | Noise added to filled pixels, on 0–1 images, as in the paper's code |
| `--road-seed` | 0 | Tie-breaking and noise |
| `--road-short-side` | 256 | Images are scaled down to this short side first; 0 keeps full size |

Each map takes about 2–5 seconds on a laptop CPU. Attribution is averaged when scaling down, and the unmodified score is measured at the same size.

## Reading the results

- Compare methods on the same images and model. Values from different models are not comparable.
- `lerf` above 1 means removing the least relevant pixels raised the score.
- Prefer several images: one image says little. On low-confidence predictions `_rel` values can become very large, so also check the raw columns.
- Scaling down matters for detectors: DETR normally sees larger images, so raise `--road-short-side` for it.

## KernelSHAP scale

`SAM_c+KernelSHAP` maps store each region's **total** contribution on every pixel of that region. BPT maps store a per-pixel share. Pixel ranking therefore treats large regions differently in the two cases, in ROAD and in the insertion/deletion AUC. Keep this in mind when comparing KernelSHAP with other methods.
