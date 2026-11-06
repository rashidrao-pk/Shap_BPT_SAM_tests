# YOLO11s

| | |
| --- | --- |
| Used with | [MS-COCO](../DATASETS/MS_COCO.md) |
| Task | Object detection, 80 COCO classes |
| Runner | `OD_yolo.py` |
| Checkpoint | `src/checkpoints/yolo11s.pt` (19 MB) |

## Download

```bash
mkdir -p src/checkpoints
curl --fail --location -o src/checkpoints/yolo11s.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

`--model yolo11s` looks in `src/notebooks/checkpoints`, `src/checkpoints`, and `checkpoints`. To use another file, pass `--weights /path/to/model.pt`.

## Why this model

- It was trained on COCO, so its classes match the annotations.
- It is a small CNN detector, compared against the transformer [DETR](DETR.md).

## Limits

- The score combines all detections of a class, so a map explains the class, not one object.
- YOLO and COCO number classes differently. Ground truth is matched by class name.
- It trained on COCO `train2017`, so do not treat [PACO](../DATASETS/PACO.md) images as unseen by it.

Usage: [YOLO guide](../XAI/DATASETS/OD_yolo.md).
