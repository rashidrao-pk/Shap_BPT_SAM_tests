# YOLO11s-OBB

| | |
| --- | --- |
| Used with | [DOTA](../DATASETS/DOTA.md) |
| Task | Oriented-box detection, 15 DOTA-v1.0 classes |
| Runner | `OD_DOTA.py` |
| Checkpoint | `src/checkpoints/yolo11s-obb.pt` (19.9 MB), SHA-256 `43fa6310…f6b1a45a` |

## Download

```bash
python src/scripts/download_datasets.py --dataset dota --assets model
```

This fetches `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt` and checks the full SHA-256. `--model yolo11m-obb` (or another OBB file) is looked up as `<model>.pt` in `src/notebooks/checkpoints`, `src/checkpoints`, and `checkpoints`; `--checkpoint /path/to/file.pt` uses that file. A detection-only (non-OBB) model is rejected.

Classes, in the model's order: plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool. The checkpoint records task `obb`, Ultralytics 8.2.100, trained from `DOTAv1-ms.yaml` (multi-scale DOTA tiles), validated on `val`.

## Why this model

- It was trained on DOTA-v1.0, so its classes match the annotations.
- It is the oriented-box sibling of the [YOLO11s](YOLO.md) used on MS-COCO.

## How it is scored

- A class's score is the **sum of the confidences of all its boxes** on the image (`--class-aggregation max` takes the largest instead), as for the COCO detectors. A map explains the class on the tile, not one object.
- Boxes under `--conf` (default 0.25, the Ultralytics and `OD_yolo` default) are dropped. As parts of the tile are masked, a box can fall under the threshold and vanish, so the score can jump instead of fading. DETR uses a threshold of 0 to avoid this; YOLO-OBB keeps 0.25 to stay comparable with `OD_yolo`.
- Images are passed in BGR order, which Ultralytics expects for NumPy arrays. Inference is at 1024 pixels, without half precision or test-time augmentation.

## Limits

- Possible leakage: whether DOTA val was in training could not be confirmed (see [DOTA](../DATASETS/DOTA.md#limits-on-what-you-can-claim)).
- Gradient methods (LRP, GradCAM, IDG, GradExp, GradShap) have no detector adapter and are reported as unsupported.
