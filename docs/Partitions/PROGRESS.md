# Check partition progress on EPITO

From the repository root, with Python and PyYAML available:

```bash
python src/scripts/check_partition_progress.py --limit 500
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
