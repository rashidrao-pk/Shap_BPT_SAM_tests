# SAM experiments after setup

On EPITO, first follow [EPITO.md](EPITO.md) to request a GPU and activate the
installed environment. Run these commands from `$PROJECT_DIR`.
One-time installation is in [EPITO_SETUP.md](EPITO_SETUP.md).

## Generate SAM1 partitions

```bash
python src/scripts/generate_sam_partitions.py \
  --model sam1 \
  --image data/sample_images/truck.jpg \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda \
  --strategy refined \
  --max-regions 62 \
  --output-dir results/partitions/sam1
```

## Generate SAM2.1 partitions

```bash
python src/scripts/generate_sam_partitions.py \
  --model sam2 \
  --image data/sample_images/truck.jpg \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --strategy refined \
  --max-regions 62 \
  --output-dir results/partitions/sam2
```

Change `--image` to run another image. Change `--max-regions` to adjust the region
budget. If GPU memory runs out, lower `--points-per-batch` (for example, to `8`).

Each run saves:

| File | Contents |
| --- | --- |
| `*_refined_62.npy` | Non-overlapping partition labels |
| `*_refined_62.png` | Preview |
| `*_refined_62.json` | Timing, parameters, and region statistics |
| `*_raw_masks.npz` | Original SAM masks |

The filename uses your selected strategy and region budget. Region labels are
partition IDs, not semantic class IDs.

## Refine saved masks without rerunning SAM

```bash
python src/scripts/generate_sam_partitions.py \
  --model sam2 \
  --image data/sample_images/truck.jpg \
  --raw-masks results/partitions/sam2/truck_sam2_raw_masks.npz \
  --strategy refined \
  --max-regions 100 \
  --output-dir results/partitions/sam2
```

## MS-COCO batches and XAI

For compatible MS-COCO batch outputs, use [Partitions/EPITO.md](../Partitions/EPITO.md).
The single-image filenames above are intended for inspection, not direct COCO
XAI discovery. After generating the batch outputs, use
[XAI/EPITO.md](../XAI/EPITO.md).

## Optional SAM3 example

Requires [SAM3 setup and checkpoint access](EPITO_SAM3.md). SAM3 uses text prompts
and a different package configuration from the SAM1/SAM2 session. Inside your
GPU allocation, activate its documented environment before this example:

```bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
```

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
mkdir -p results/partitions/sam3

```

```bash
python - <<'PY'
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

out_dir = Path("results/partitions/sam3")
out_dir.mkdir(parents=True, exist_ok=True)

image = Image.open("data/sample_images/truck.jpg").convert("RGB")

print("[1/4] Loading SAM3...")
model = build_sam3_image_model()
processor = Sam3Processor(model)

print("[2/4] Processing image...")
with torch.inference_mode():
    state = processor.set_image(image)

    print("[3/4] Running text-conditioned segmentation...")
    output = processor.set_text_prompt(
        state=state,
        prompt="pickup truck"
    )

masks = output["masks"].detach().cpu().numpy()
scores = output["scores"].detach().cpu().numpy()

np.savez_compressed(
    out_dir / "pickup_truck.npz",
    masks=masks,
    scores=scores,
)

print("[4/4] Results saved")
print("Instances:", len(scores))
print("Masks shape:", masks.shape)
print("Scores:", scores)
print("Output:", out_dir / "pickup_truck.npz")
PY
```

This saves text-conditioned masks and scores, rather than a disjoint partition map.
