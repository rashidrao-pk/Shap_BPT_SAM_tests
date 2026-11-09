# DOTA

[DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) is an aerial-image detection dataset: 15 object classes, each object marked by an oriented (rotated) box. It adds a remote-sensing domain, where objects are small, numerous and seen from above.

| | |
| --- | --- |
| Images | val: 458 source images, cut here into 1024×1024 tiles; 300 tiles kept by default |
| Ground truth | One oriented box per object, filled as a polygon; 15 classes |
| Model | [YOLO11s-OBB](../MODELS/YOLO_OBB.md) |
| Download | `DOTAv1.zip`, 1.99 GB (all splits); SHA-256 pinned |
| Licence | Academic use only, commercial use prohibited; Google Earth images also under the Google Earth terms ([DOTA terms](https://captain-whu.github.io/DOTA/dataset.html)) |

Classes: plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool.

## 1. Download the dataset and cut tiles

```bash
python src/scripts/download_datasets.py --dataset dota --assets all
```

This downloads Ultralytics' `DOTAv1.zip` to `/beegfs/home/mrashid/datasets/DOTA/raw` (checked against its SHA-256), extracts it, cuts the val images into tiles, and downloads the detector. Only `--split val` is accepted: test labels are withheld by the DOTA authors, and the detector was trained on train.

```text
DOTA/prepared_1024/val/images/<tile-id>.png   # selected tiles
DOTA/prepared_1024/val/labels/<tile-id>.txt   # objects in tile pixels
DOTA/prepared_1024/val/tile_index.csv         # every candidate tile, its source and whether it was kept
```

Tile IDs are `<source image>__<x0>_<y0>`, for example `P0003__123_0`.

### How tiles are made

| Option | Default | Meaning |
| --- | --- | --- |
| `--tile` | 1024 | Tile side. Source images reach thousands of pixels; 1024 is SAM's input size and the detector's training size |
| `--stride` | 1024 | No overlap, so no object is explained twice. The last window moves inward so every tile is full size |
| `--min-keep` | 0.7 | Share of an object's area inside the tile needed to count it |
| `--min-objects` | 3 | Counted objects a tile needs to be a candidate |
| `--max-tiles` | 300 | Tiles kept, a seeded random sample of the candidates |
| `--seed` | 0 | Sample seed |
| `--label-format` | `auto` | `dota-txt` (official labels with the difficult flag) when present, else `yolo-obb` |

- **Labels.** The zip holds two copies of the val labels. `labels/val_original` is the official DOTA labelTxt, in pixels, with a *difficult* flag per object; `labels/val` is Ultralytics' normalised copy without it. The tiler reads the official one.
- **Ignored regions.** An object counts as ground truth when at least 70% of it lies inside the tile and DOTA does not flag it as difficult. Other objects in the tile are written with `difficult=1`: they are left out of the GT mask and out of every score, so a half-cut plane the detector can still see is not a false localisation.
- **Padding.** Where a source image is smaller than a tile, the tile is padded with black at the right or bottom. Padding has no objects.
- **Label file.** One object per line: `x1 y1 x2 y2 x3 y3 x4 y4 <class> <difficult 0|1>`, in tile pixels. The class name may contain spaces, so lines are read from the right. Corners keep their position even outside the tile; filling the polygon inside the tile gives the cut part exactly.
- **Repeat runs** skip existing tiles and leave identical files untouched.

Changing `--tile`, `--stride`, `--min-keep` or the sample gives different tiles under the same folder names. Use a new `--root` or `output.partitions` root for that.

## 2. Models

- [YOLO11s-OBB](../MODELS/YOLO_OBB.md), downloaded by the same command, or alone with `--assets model`.

## 3. Next steps

1. [Generate partitions](../Partitions/DATASETS/DOTA.md)
2. [Evaluate partitions against GT](../Partitions/EVALUATE_PARTITIONS.md)
3. [Run explanations](../XAI/DATASETS/DOTA.md)

## Limits on what you can claim

- **Possible leakage.** The detector was trained on DOTAv1 by Ultralytics, whose headline accuracy is measured on the DOTA test server; detectors scored that way are usually trained on train and val. The weights record their training file only as `DOTAv1-ms.yaml`, so whether val was used could not be confirmed. Treat results as a benchmark on familiar data, not on unseen data.
- **A box is not an outline.** GT is the filled oriented box, which also covers ground around a ship or a plane's wings. IoU and pointing scores are capped by the box shape.
- **Tiny objects.** Many vehicles and ships cover a few hundred pixels of a 1024² tile. Partitions with at most 62 regions cannot isolate each one; see the partition verdict before drawing conclusions.
- **Domain shift.** SAM was trained on natural photographs, not aerial images. How well its partitions fit this domain is exactly what the partition evaluation measures.
- **Tiles are not independent.** Several tiles come from one source image; `tile_index.csv` lists the source of each. Report the number of source images next to the number of tiles.
- **Ignored regions** are border-cut objects (under 70% inside) and objects DOTA marks as difficult.
- Research diagnostics only.
