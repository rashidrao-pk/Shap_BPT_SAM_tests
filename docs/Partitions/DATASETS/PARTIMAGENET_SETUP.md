# PartImageNet: one-time setup

Use the existing [EPITO environment setup](../full/EPITO_SETUP.md). The [authors' release](https://github.com/TACJu/PartImageNet) provides Seg and OOD splits with COCO-style part annotations. We default to **Seg**, where train/val/test share object classes; OOD separates classes across splits.

## Download dataset and classifier

On a node with internet access, from the repository root:

```bash
python src/scripts/download_partimagenet.py
```

This downloads and validates `PartImageNet_Seg.zip`, extracts under `/beegfs/home/mrashid/datasets/PartImageNet`, and downloads `checkpoints/partimagenet/resnet50-11ad3fa6.pth` with a SHA256-prefix check. The [official archive](https://huggingface.co/datasets/turkeyju/PartImageNet/tree/main) is about 3.12 GB compressed; allow additional space for extraction and generated partitions.

The download script uses Python's standard library: it does not import NumPy, PyTorch, Kagglehub, or Xet. Existing downloaded files are reused. Invalid ZIPs produce an error before extraction; replace damaged archives with a fresh download.

Optional separate downloads:

```bash
python src/scripts/download_partimagenet.py --assets dataset
python src/scripts/download_partimagenet.py --assets model
```

Use `--variant OOD` only if you specifically want the authors' OOD split. Both variants can contain images used to train the ImageNet classifier; these explanations do not establish independent classification accuracy.

## Configure one split

Inspect the extracted paths:

```bash
find /beegfs/home/mrashid/datasets/PartImageNet -type f -name '*.json'
find /beegfs/home/mrashid/datasets/PartImageNet -maxdepth 3 -type d
```

Edit `src/configs/PartImageNet_epito.yaml`:

- `data.image_dir`: directory containing the selected split's image files, or its parent. Subdirectories are searched when necessary.
- `data.annotation_file`: the chosen split's COCO part-annotation JSON, for example validation annotations. Use the extracted filename, not an assumed filename.
- `data.ground_truth.annotation_file`: the same JSON.
- `data.ground_truth.root`: an existing dataset directory.

The initial config paths are examples until checked against your extracted archive. All images in the chosen JSON must resolve uniquely under `image_dir`; missing or ambiguous paths are errors.

The classifier is Torchvision's [ImageNet ResNet50 V2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), with its original ImageNet class labels and preprocessing. It is not a trained part-segmentation model. SAM1/SAM2 checkpoints are separate; their download commands are in [SAM checkpoint setup](BUSI_SETUP.md#3-sam-checkpoint).

Next: [partition generation and preview](PARTNET.md), then [XAI](../../XAI/PARTIMAGENET.md). Downloads run without scientific packages; partition generation and XAI require the working GPU environment.
