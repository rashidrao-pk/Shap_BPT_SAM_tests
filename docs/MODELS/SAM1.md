# SAM-1

Generates the image partitions. It is not the model being explained.

| | |
| --- | --- |
| Used by | Partition generation for every dataset |
| Architecture | Segment Anything, ViT-B |
| Checkpoint | `checkpoints/sam1/sam_vit_b_01ec64.pth` (375 MB) |

## Download

```bash
mkdir -p checkpoints/sam1
curl -L --fail \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o checkpoints/sam1/sam_vit_b_01ec64.pth
```

For the package installation, see [EPITO setup](../Partitions/full/EPITO_SETUP.md).

## Use

```bash
--model sam1 --sam1-variant vit_b \
--checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth"
```

Partition folders are named `<Dataset>_sam1_vit_b_62_compact-sam_<masks>`.
