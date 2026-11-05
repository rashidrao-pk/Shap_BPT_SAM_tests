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

The archive's internal folders need not be named `PartImageNet_Seg`. Automatically locate the extracted validation part annotations, validate their image paths, and update the config:

```bash
python src/scripts/download_partimagenet.py --assets configure --split val
```

This step uses the existing extraction and performs **no downloads**. It sets `data.image_dir` to the extraction root and updates both annotation paths together, retaining output paths and saving the original config as `.yaml.bak`. It uses the already-installed PyYAML package, with no NumPy/PyTorch imports.

If multiple annotation JSONs match, it stops without changing the config. Select the correct part annotation explicitly:

```bash
python src/scripts/download_partimagenet.py --assets configure \
  --annotation-file /absolute/path/to/validation_part_annotations.json
```

Confirm selection before loading SAM:

```bash
python src/scripts/generate_partimagenet_partitions.py \
  --config PartImageNet_epito --dry-run --limit 2
```

For a first-time download plus configuration, use `python src/scripts/download_partimagenet.py --configure`. To inspect or set paths manually, use the following steps.

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

### Both `val` and `val_whole` are present

Use `PartImageNet/annotations/val/val.json` for part annotations. The parallel `val_whole` directory contains whole-object annotations. Automatic setup excludes `_whole` paths. With an older script, select the part JSON explicitly:

```bash
python src/scripts/download_partimagenet.py --assets configure --split val \
  --annotation-file /beegfs/home/mrashid/datasets/PartImageNet/PartImageNet/annotations/val/val.json
```
