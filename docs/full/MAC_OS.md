## SETUP in MAC-OS

```bash
git clone https://github.com/rashidrao-pk/Shap_BPT_SAM_tests.git
cd Shap_BPT_SAM_tests
```

## 1. Mac: create the environment

```bash
conda create -n shapbpt-sam python=3.12 -y
conda activate shapbpt-sam

which python
python --version

# Upgrade packaging tools:
python -m pip install --upgrade pip setuptools wheel
```

## 2. Mac: install PyTorch

- On macOS, install the normal PyTorch wheels:

```bash
pip install torch torchvision

# Check
python - <<'PY'
import platform
import torch

print("Platform:", platform.platform())
print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    print("Selected device: mps")
else:
    print("Selected device: cpu")
PY
```

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 3. Install your ShapBPT repo

```bash
cd ~/repos/Shap_BPT_SAM_tests
# install your package editable:
pip install -e .

pip install PyYAML pycocotools ultralytics scipy pandas opencv-python jupyter ipykernel
```

### Verify ShapBPT:

```bash
python - <<'PY'
import shap_bpt
print("shap_bpt imported successfully")
print(shap_bpt.__file__)
PY
```

## 4. Mac: install SAM 1

- The official SAM1 repository still supports installation directly from GitHub.
  Install:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

#### Then verify:

```bash
python - <<'PY'
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)

print("SAM1 installation OK")
print("Available models:", list(sam_model_registry.keys()))
PY
```

#### Download the ViT-B checkpoint:

```bash
mkdir -p checkpoints/sam1

curl -L \
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
-o checkpoints/sam1/sam_vit_b_01ec64.pth

ls -lh checkpoints/sam1/
```

## 5. Mac: install SAM 2.1

```bash
cd ~/repos
mkdir -p checkpoints/third_party

cd checkpoints/third_party

git clone https://github.com/facebookresearch/sam2.git
cd sam2

```

#### On Mac we specifically disable compilation of the CUDA extension:

```bash
# SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
SAM2_BUILD_CUDA=0 python -m pip install -e .

# VERIFY

python - <<'PY'
import torch
import sam2

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

print("SAM2 installation OK")
print("SAM2:", sam2.__file__)
print("torch:", torch.__version__)
print("MPS:", torch.backends.mps.is_available())
PY
```

## 6. Download SAM2.1 checkpoints

#### 1. Download the SAM2.1 checkpoints:

```bash
~/repos/third_party/sam2
```

```bash
cd checkpoints
./download_ckpts.sh
ls -lh
cd ..
```

#### 2. Test loading the model on MPS

Go back to your SAM2 directory:

```bash
cd ..
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Execute:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python - <<'PY'
import torch
from sam2.build_sam import build_sam2

device = torch.device("mps")

model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    "checkpoints/sam2.1_hiera_tiny.pt",
    device=device,
    apply_postprocessing=False,
)

print("=" * 50)
print("SAM2.1 Tiny loaded successfully!")
print("Device:", device)
print("Model:", type(model).__name__)
print("=" * 50)
PY
```

## Step 2 — Generate the first SAM2 partitions

From your current sam2 directory, run:

```bash
mkdir -p ../../data/sample_images
mkdir -p ../../results/sam2
```

Download one test image:

```bash
curl -L \
  https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg \
  -o ../../data/sample_images/truck.jpg
```

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python - <<'PY'
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

root = Path.cwd().parents[1]
image_path = root / "data/sample_images/truck.jpg"
output_dir = root / "results/sam2"
output_dir.mkdir(parents=True, exist_ok=True)

image = np.array(Image.open(image_path).convert("RGB"))

print("Input image:", image.shape)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    "checkpoints/sam2.1_hiera_tiny.pt",
    device=device,
    apply_postprocessing=False,
)

generator = SAM2AutomaticMaskGenerator(
    model=model,
    points_per_side=16,
    points_per_batch=16,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.85,
    crop_n_layers=0,
    output_mode="binary_mask",
)

print("Generating SAM2 masks...")
start = time.perf_counter()

with torch.inference_mode():
    masks = generator.generate(image)

elapsed = time.perf_counter() - start

print("Generated masks:", len(masks))
print("Inference time (seconds):", round(elapsed, 2))

# Convert overlapping instance masks into one non-overlapping partition map.
# Larger masks first; smaller masks may overwrite overlapping pixels.
h, w = image.shape[:2]
partitions = np.zeros((h, w), dtype=np.uint16)

sorted_masks = sorted(
    masks,
    key=lambda m: m["area"],
    reverse=True,
)

for label, mask in enumerate(sorted_masks, start=1):
    partitions[mask["segmentation"]] = label

# Renumber visible labels consecutively.
_, partitions = np.unique(partitions, return_inverse=True)
partitions = partitions.reshape(h, w).astype(np.uint16)

np.save(output_dir / "truck_sam2_partitions.npy", partitions)

# Preserve the original SAM masks as well.
np.savez_compressed(
    output_dir / "truck_sam2_raw_masks.npz",
    masks=np.stack(
        [m["segmentation"] for m in masks],
        axis=0
    ) if masks else np.zeros((0, h, w), dtype=bool),
)

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image)
axes[0].set_title("Original image")

axes[1].imshow(partitions, cmap="nipy_spectral", interpolation="nearest")
axes[1].set_title(f"SAM2 partitions ({len(np.unique(partitions))} regions)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(output_dir / "truck_sam2_partitions.png", dpi=200)
plt.close()

print("Partition labels:", len(np.unique(partitions)))
print("Output directory:", output_dir)
print("SAM2 partition generation completed.")
PY
```

## TEST SAM on MAC

### SAM 1

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

python src/scripts/generate_sam_partitions.py \
  --model sam1 \
  --image data/sample_images/truck.jpg \
  --device mps \
  --max-regions 62 \
  --strategy refined \
  --output-dir results/partitions/sam1
```

### SAM2

```bash
python src/scripts/generate_sam_partitions.py \
  --model sam2 \
  --image data/sample_images/truck.jpg \
  --device mps \
  --max-regions 62 \
  --strategy refined \
  --output-dir results/partitions/sam2
```
