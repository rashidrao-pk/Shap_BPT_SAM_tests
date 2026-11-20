# Check partition progress on EPITO

From the repository root, with Python and PyYAML available:

```bash
env=/beegfs/home/mrashid/pt_312_sam_x86
source "$env/bin/activate"
```

```bash
python src/scripts/check_partition_progress.py --limit 1000  --gt-only
```

This checks MS-COCO, ImageNet, CelebA-HQ, and CAMO using their EPITO configs.
It reports a row for each discovered dataset/model folder, plus the configured
`mask_dir_final` if that folder has not been created yet. No GPU, SAM, or image
decoding is needed. Input image paths and annotation files must be accessible.

The table shows completed images, total dataset images, percentage, partial
images, remaining images, and partition folder name. Totals come from the COCO
annotation image list or supported image files in each directory config.
Different model folders are separate experiments; do not add their counts to
get a unique dataset total.

An image is **completed** when its `*_results.json` is readable and all stages
recorded in `saved_partitions` exist. Older metadata falls back to its recorded
stage selection, or `sam` + `refined`. Files without complete metadata/stages are
partial. The check counts file existence; it does not verify array contents or
whether a raw-mask target was reached. Remaining includes partial images.

## Require specific stages

```bash
python src/scripts/check_partition_progress.py \
  --methods sam compact boosted filled boosted_filled refined
```

This checks whether every image has all requested stages, rather than just the
stages each run selected. It can reveal datasets generated with fewer stages.

## Check selected datasets or a limited run

```bash
python src/scripts/check_partition_progress.py --config ImageNet_epito CAMO_epito
python src/scripts/check_partition_progress.py --config MSCOCO_epito --limit 500
```

By default progress is against the full dataset. Use `--limit 500` when measuring
a generation command that used `--limit 500`; the denominator and image ordering
then match that subset. With several configs, the limit applies to each dataset.

To check only each config's selected folder:

```bash
python src/scripts/check_partition_progress.py --configured-only
```

## Save a report

```bash
python src/scripts/check_partition_progress.py --json results/partition_progress.json
```

Rerun the command to refresh progress. It is read-only and can run while GPU jobs
continue. A currently written image can appear partial until its final metadata
has been saved. Errors for inaccessible datasets are printed without hiding
results for the others; the command exits nonzero if any dataset could not be read.

## Storage and generation time

The same command now includes storage and time automatically:

```bash
python src/scripts/check_partition_progress.py --limit 1000

python src/scripts/check_partition_progress.py --limit 1000 --gt-only --no-busi-pipeline
```

**ImageNet and `--gt-only`.** ImageNet partitions are generated with `--gt-only`: only validation images with an
ImageNet-S mask, chosen before `--limit`. Counting the first 1000 of _all_ validation images would find only the
few of them that have masks (e.g. `11 / 1000`, while `Size(folder)` and `Time(folder)` show the full run). The
check therefore also counts the GT-only selection and uses it when it matches the folders better; the output then
says `counted on the GT-only selection`. `--gt-only` forces it, `--all-inputs` turns the automatic switch off.

- `Size(subset)`: saved partition arrays, raw-mask NPZs, and results JSONs for the selected input IDs, including partial outputs.
- `Size(folder)`: all files recursively in that partition folder, including images beyond the limit and any extra files.
- `Time(subset)`: sum of `total_time_sec` for completed images in the selected subset.
- `Time(folder)`: sum of `total_time_sec` for all completed images in that folder.
- `Greedy`, `Split`, `Other`: images of the subset with `<id>_compact_greedy.npy`, `<id>_compact_split.npy` or another compact file (e.g. `mso_0.5`). `*` marks the one XAI reads (`data.compact_strategy`, split by default). A line under the row warns when it lags behind `Completed`.

Times use `HH:MM:SS`. Missing timing records are reported, and no timing data is
shown as `n/a`. Timings count recorded successful per-image generation work;
they do not include Slurm queue time, model startup, failed runs, or previous
outputs overwritten by another run. They are not elapsed job wall time.

Storage is logical file size (not filesystem allocated blocks). The matching
sibling `.log` and original dataset images are excluded. `--methods` also
controls which images qualify as complete for the timing sums; storage still
includes every saved stage. Different folders can contain the same input images.
The JSON report includes exact byte sizes, timing sums, and timing sample counts.

### Time summary

After the table, the command prints the `Time(subset)` sums grouped three ways:

- **By model:** `SAM1-VB` (SAM-1 ViT-B), `SAM2-T` (tiny), `SAM2-L` (large), read from the folder names.
- **By dataset:** all of that dataset's folders together.
- **Total:** every folder listed, also in hours.

`s/image` is the mean recorded time per image. It depends on the dataset (image size, mask target such as `sam_200` or `sam_50`), so compare models within one dataset rather than across the by-model totals. With `--json`, the same numbers are saved under `time_summary`.
