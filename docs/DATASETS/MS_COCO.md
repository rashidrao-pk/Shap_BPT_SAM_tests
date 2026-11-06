# MS-COCO

[COCO 2017](https://cocodataset.org/#download) provides everyday scenes with instance masks for 80 object classes. Here it is the detection benchmark: the explained model is a detector, and the mask marks the explained class.

| | |
| --- | --- |
| Images | val2017: 5,000 images, 4,952 of them annotated |
| Ground truth | 36,781 instance annotations, 80 classes |
| Models | [YOLO11s](../MODELS/YOLO.md), [DETR](../MODELS/DETR.md) |
| Download | 815 MB images, 253 MB annotations |

## 1. Download the dataset

Only val2017 and its annotations are needed. The layout below matches `MSCOCO_epito.yaml`:

```bash
ROOT=/beegfs/home/mrashid/datasets/MS_COCO
mkdir -p "$ROOT/annotations/annotations_trainval2017"
curl --fail --location --retry 3 -o "$ROOT/val2017.zip" \
  http://images.cocodataset.org/zips/val2017.zip
curl --fail --location --retry 3 -o "$ROOT/annotations_trainval2017.zip" \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q "$ROOT/val2017.zip" -d "$ROOT"
unzip -q "$ROOT/annotations_trainval2017.zip" -d "$ROOT/annotations/annotations_trainval2017"
```

```text
MS_COCO/
├── val2017/000000000139.jpg
└── annotations/annotations_trainval2017/annotations/instances_val2017.json
```

Check that `val2017` holds 5,000 files. Your local copy at `/Users/rashid/data/DS/MS_COCO` uses this layout, and `MSCOCO_mac.yaml` points to it.

## 2. Models

- [YOLO11s](../MODELS/YOLO.md)
- [DETR-ResNet-50](../MODELS/DETR.md)

## Next steps

1. [Generate partitions](../Partitions/DATASETS/MSCOCO_EPITO.md)
2. [Run explanations](../XAI/DATASETS/MSCOCO.md)
