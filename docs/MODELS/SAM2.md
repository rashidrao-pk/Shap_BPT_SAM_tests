# SAM-2

Generates the image partitions. It is not the model being explained.

| | |
| --- | --- |
| Used by | Partition generation for every dataset |
| Architecture | SAM 2.1 Hiera |
| Checkpoints | `$SAM2_DIR/checkpoints/` |

| Size | Used for | Checkpoint | Config |
| --- | --- | --- | --- |
| Tiny (156 MB) | Default | `sam2.1_hiera_tiny.pt` | `configs/sam2.1/sam2.1_hiera_t.yaml` |
| Large (898 MB) | MS-COCO | `sam2.1_hiera_large.pt` | `configs/sam2.1/sam2.1_hiera_l.yaml` |

## Download

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
mkdir -p "$SAM2_DIR/checkpoints"
curl -L --fail \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt"
curl -L --fail \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -o "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt"
```

For the package installation, see [EPITO setup](../Partitions/full/EPITO_SETUP.md).

## Use

```bash
--model sam2 \
--checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
--sam2-config configs/sam2.1/sam2.1_hiera_t.yaml
```

Partition folders are named `<Dataset>_sam2_tiny_62_compact-sam_<masks>`. The checkpoint and config must be the same size.
