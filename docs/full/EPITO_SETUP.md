# SETUP on EPITO (Linux / NVIDIA CUDA)

For later sessions after installation, use [EPITO.md](EPITO.md).

This guide follows [MAC_OS.md](MAC_OS.md), using the EPITO paths and environment
recorded in [../README.md](../README.md) and `src/scripts/prepare_setup.sh`.
Run the steps in order in the same shell. Adjust the username/paths for another
account. These commands have not been executed on EPITO from this workspace.

## 1. Connect and request an A100 GPU

From your local terminal:

```bash
ssh epito-mercurio
```

On EPITO:

```bash
srun -p epito --gres=gpu:a100:1 -J "ShapBPT SAM setup" --time=06:00:00 --pty bash
hostname
nvidia-smi
export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
export SAM2_DIR=/beegfs/home/mrashid/repos/XAI/third_party/sam2
```

Run model checks and mask generation inside this allocation. If compute nodes
cannot reach the internet, clone repositories and download checkpoints on a
network-enabled login node using these shared-storage paths.

## 2. Copy `pt_312` into a separate SAM environment

Keep the original `/beegfs/home/mrashid/pt_312` environment for existing
experiments. Create `/beegfs/home/mrashid/pt_312_sam` as a copy before installing
SAM dependencies. The source environment must already exist.

Use `virtualenv-clone` so activation scripts and executable paths are updated
for the destination. Install the cloning tool in a temporary environment to
leave `pt_312` untouched. Run once; the destination must not already exist:

```bash
export SAM_ENV=/beegfs/home/mrashid/pt_312_sam
CLONE_TOOLS_DIR=$(mktemp -d /tmp/pt312-clone-tools.XXXXXX)
python3.12 -m venv "$CLONE_TOOLS_DIR"
"$CLONE_TOOLS_DIR/bin/python" -m pip install virtualenv-clone
"$CLONE_TOOLS_DIR/bin/virtualenv-clone" \
  /beegfs/home/mrashid/pt_312 "$SAM_ENV"
```

Activate the copy for every installation and experiment in this guide:

```bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
# Use the same site package selection as the working EPITO setup.
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages
which python
python --version
python -m pip --version

python - <<'PY'
import torch
import torchvision
print("PyTorch:", torch.__version__, torch.__file__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Check the Slurm allocation and NVIDIA driver"
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA calculation:", (torch.ones(4, device="cuda") + 1).cpu())
PY
```

The copy has its own installation directory, so `python -m pip install` writes
to `pt_312_sam`. However, this `PYTHONPATH` gives shared site packages precedence
where names overlap. Editable packages and external data/checkpoints may also
refer to shared locations. This is a separate installation environment, but it
still depends on EPITO's shared software stack.

### If torchvision reports `operator torchvision::nms does not exist`

This usually indicates incompatible torch/torchvision builds or a torchvision
compiled extension that could not load. A cloned environment can contain local
copies that conflict with the site stack. Restore the original EPITO path
selection and test in a fresh Python process:

```bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages
python - <<'PY'
import sys
import torch
print("Interpreter:", sys.executable)
print("Environment prefix:", sys.prefix)
print("PyTorch:", torch.__version__, torch.__file__)
print("CUDA runtime:", torch.version.cuda)
import torchvision
print("torchvision:", torchvision.__version__, torchvision.__file__)
from torchvision.ops import nms
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]])
print("NMS CPU check:", nms(boxes, torch.tensor([0.9, 0.8]), 0.5))
assert torch.cuda.is_available(), "Request a Slurm GPU allocation first"
print("GPU:", torch.cuda.get_device_name(0))
print("NMS CUDA check:", nms(boxes.cuda(), torch.tensor([0.9, 0.8], device="cuda"), 0.5))
PY
```

If a previous version of this guide created `epito-pytorch.pth`, the explicit
`PYTHONPATH` above takes precedence; recreating the environment is unnecessary.
A prompt displaying `(pt_312)` can be inherited from the clone; `sys.executable`
and `sys.prefix` identify the actual environment.

If this still fails, retain the printed versions/paths and the full error to
check whether the shared stack itself works. Do not install arbitrary torch or
torchvision versions over it. For a fully self-contained environment, omit the
shared `PYTHONPATH` and remove the earlier `epito-pytorch.pth` entry, then install
a matched CUDA wheel pair using the
[official PyTorch version table](https://pytorch.org/get-started/previous-versions/)
and a CUDA build supported by EPITO's driver.

## 3. Install ShapBPT and dependencies

Clone only if the checkout does not already exist:

```bash
mkdir -p /beegfs/home/mrashid/repos/XAI
cd /beegfs/home/mrashid/repos/XAI
git clone https://github.com/rashidrao-pk/Shap_BPT_SAM_tests.git
```

```bash
cd "$PROJECT_DIR"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install PyYAML pycocotools ultralytics scipy pandas opencv-python pillow jupyter ipykernel
# Optional dependencies for DETR experiments:
python -m pip install transformers timm

python - <<'PY'
import shap_bpt
from shap_bpt import bpt
print("shap_bpt:", shap_bpt.__file__)
print("Compiled BPT extension:", bpt.__file__)
PY
```

The editable installation compiles the Cython BPT extension and needs a working
C/C++ compiler. Load EPITO's compiler module if necessary. Dependencies are
listed explicitly because the root `requirements.txt` contains the inline entry
`ultralytics --no-deps`, which is not a valid ordinary requirements-file entry.

## 4. Install SAM 1 and download ViT-B

```bash
cd "$PROJECT_DIR"
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
python - <<'PY'
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
print("SAM1 installation OK; models:", list(sam_model_registry))
PY

mkdir -p "$PROJECT_DIR/checkpoints/sam1"
curl -fL --retry 3 \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth"

python - <<'PY'
import os
from pathlib import Path
import torch
from segment_anything import sam_model_registry
assert torch.cuda.is_available(), "Run inside the Slurm GPU allocation"
checkpoint = Path(os.environ["PROJECT_DIR"]) / "checkpoints/sam1/sam_vit_b_01ec64.pth"
model = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to("cuda").eval()
print("SAM1 ViT-B loaded:", next(model.parameters()).device)
PY
```

## 5. Install SAM 2 for SAM 2.1 checkpoints

SAM2.1 uses the official `sam2` package. The upstream
[installation guide](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)
specifies Python >= 3.10 and PyTorch >= 2.5.1 with matching torchvision.

```bash
mkdir -p /beegfs/home/mrashid/repos/XAI/third_party
cd /beegfs/home/mrashid/repos/XAI/third_party
# Skip the clone if this checkout already exists.
git clone https://github.com/facebookresearch/sam2.git
cd "$SAM2_DIR"
SAM2_BUILD_CUDA=0 python -m pip install -e .

python - <<'PY'
import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
print("SAM2 installation OK:", sam2.__file__)
PY
```

This skips the optional CUDA extension while retaining GPU inference. Extension
postprocessing for small holes/sprinkles is unavailable; the examples also set
`apply_postprocessing=False` and leave mask-generator small-region removal off.

To build the extension instead, load EPITO's CUDA toolkit matching
`torch.version.cuda`, check `nvcc --version`, and run from `$SAM2_DIR`:

```bash
SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 python -m pip install -v --no-build-isolation -e .
```

`nvidia-smi` shows driver capability; `nvcc` shows the compiler toolkit version.
PyTorch's CUDA runtime alone does not supply `nvcc`. Consult the upstream
[extension troubleshooting](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)
if this optional build fails.

## 6. Download SAM2.1 checkpoints and verify CUDA loading

Download Tiny, used in this guide:

```bash
mkdir -p "$SAM2_DIR/checkpoints"
curl -fL --retry 3 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt"
ls -lh "$SAM2_DIR/checkpoints"
```

Alternatively, download all SAM2.1 sizes with the upstream script:

```bash
cd "$SAM2_DIR/checkpoints"
bash download_ckpts.sh
```

Match each checkpoint with its model-size/release config from the
[official checkpoint list](https://github.com/facebookresearch/sam2#download-checkpoints).
Run model code from the ShapBPT root:

```bash
cd "$PROJECT_DIR"
python - <<'PY'
import os
from pathlib import Path
import torch
from sam2.build_sam import build_sam2
assert torch.cuda.is_available(), "Run inside the Slurm GPU allocation"
model = build_sam2(
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    str(Path(os.environ["SAM2_DIR"]) / "checkpoints/sam2.1_hiera_tiny.pt"),
    device="cuda",
    apply_postprocessing=False,
)
print("SAM2.1 Tiny loaded:", type(model).__name__)
print("Device:", next(model.parameters()).device)
PY
```

## 7. Generate the first SAM2 partitions

```bash
cd "$PROJECT_DIR"
mkdir -p data/sample_images results/sam2
curl -fL --retry 3 \
  https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg \
  -o "$PROJECT_DIR/data/sample_images/truck.jpg"
```

Save raw masks, a non-overlapping partition map, and a headless PNG preview.
Uncovered pixels form a background region; visible labels are renumbered.

python - <<'PY'
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import os
root = Path(os.environ["PROJECT_DIR"])
image_path = root / "data/sample_images/truck.jpg"
output_dir = root / "results/sam2"
output_dir.mkdir(parents=True, exist_ok=True)

image = np.array(Image.open(image_path).convert("RGB"))

print("Input image:", image.shape)

assert torch.cuda.is_available(), "Request a Slurm GPU allocation first"
device = "cuda"

model = build_sam2(
"configs/sam2.1/sam2.1_hiera_t.yaml",
str(Path(os.environ["SAM2_DIR"]) / "checkpoints/sam2.1_hiera_tiny.pt"),
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

\_, partitions = np.unique(partitions, return_inverse=True)
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

import matplotlib
matplotlib.use("Agg")
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

````

## 8. Resume EPITO experiments

In a new terminal, request a GPU and activate the SAM copy:

```bash
srun -p epito --gres=gpu:a100:1 -J "ShapBPT SAM" --time=06:00:00 --pty bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages
export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
export SAM2_DIR=/beegfs/home/mrashid/repos/XAI/third_party/sam2
cd "$PROJECT_DIR"
````

The existing
`prepare_setup.sh` helper activates the original `pt_312` environment, so use
the commands above for the separate SAM environment.

Check dataset, annotation, partition, and output paths in
[MSCOCO_epito.yaml](../../src/configs/MSCOCO_epito.yaml) before full runs.
The truck example only writes `results/sam2`; it does not prepare the MS-COCO
partition dataset required by that configuration. Follow the
[EPITO experiment guide](../README_exp_epito.md) after preparing those inputs.

Small YOLO smoke run with existing dataset and partition files:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
python src/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --max-evals 50 \
  --eval-batch-size 16 \
  --auc-batch-size 8 \
  --verbose-level medium \
  --model yolo11s \
  --bg-type noise \
  --class-aggregation sum
```

For CUDA out-of-memory errors, reduce batch sizes or SAM's `points_per_batch`.
For headless OpenCV `libGL.so.1` import errors, replace `opencv-python` with
`opencv-python-headless`; avoid keeping both installed.

Record the environment and SAM2 revision after the partition example:

```bash
cd "$PROJECT_DIR"
python -m pip freeze > results/sam2/pip-freeze-epito.txt
git -C "$SAM2_DIR" rev-parse HEAD > results/sam2/sam2-revision.txt
```
