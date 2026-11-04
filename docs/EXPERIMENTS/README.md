## Model Families

| Task             | Model                            |                        Output classes | Script                                |
| :--------------- | :------------------------------- | ------------------------------------: | :------------------------------------ |
| Object detection | YOLOv11 / custom YOLO checkpoint | Usually 80 for COCO, 26 for `yolo_26` | `examples/scripts/run_yolo_full.py`   |
| Classification   | ResNet-50 ImageNet               |                                  1000 | `examples/scripts/run_resnet_full.py` |
| Classification   | ViT-B/16 ImageNet                |                                  1000 | `examples/scripts/run_vit_full.py`    |
| Object detection | DETR ResNet-50                   |            COCO-style detector labels | `examples/scripts/run_detr_full.py`   |
