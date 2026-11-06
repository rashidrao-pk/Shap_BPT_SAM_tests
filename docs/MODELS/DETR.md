# DETR-ResNet-50

| | |
| --- | --- |
| Used with | [MS-COCO](../DATASETS/MS_COCO.md) |
| Task | Object detection, COCO classes |
| Runner | `OD_DETR.py` |
| Checkpoint | [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50), Hugging Face cache |

## Download

Needs `transformers` and `timm`. Run this on a node with internet access, so GPU jobs can use `--local-files-only`:

```bash
python - <<'PY'
from transformers import DetrForObjectDetection, DetrImageProcessor
name = 'facebook/detr-resnet-50'
DetrImageProcessor.from_pretrained(name)
DetrForObjectDetection.from_pretrained(name)
PY
```

## Why this model

- It was trained on COCO, so its classes match the annotations.
- It is a transformer detector, compared against the CNN [YOLO11s](YOLO.md).

## Limits

- `--threshold` defaults to 0, which keeps masked scores continuous. Displayed boxes use at least 0.3.
- `--class-aggregation sum` adds all detections of the class. `max` uses the strongest one.
- It trained on COCO `train2017`, so do not treat [PACO](../DATASETS/PACO.md) images as unseen by it.

Usage: [MS-COCO XAI](../XAI/DATASETS/MSCOCO.md).
