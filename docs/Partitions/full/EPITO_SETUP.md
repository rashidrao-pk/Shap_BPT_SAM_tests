# SETUP on EPITO (Linux / NVIDIA CUDA)

For later sessions after installation, use [EPITO.md](EPITO.md).

This guide follows [MAC_OS.md](MAC_OS.md), using the EPITO paths and environment
recorded in [../README.md](../README.md) and `src/scripts/prepare_setup.sh`.
Run the steps in order in the same shell. Adjust the username/paths for another
account. The environment has been set up on EPITO; use this document when rebuilding it.

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

If torchvision reports `operator torchvision::nms does not exist`, restore the
`PYTHONPATH` shown above and retry in a fresh Python process. Check
`torch.__file__` and `torchvision.__file__` to confirm which packages are loaded.
A cloned prompt may still display `(pt_312)`; `which python` should point to
`pt_312_sam/bin/python`.

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

## 7. Download the sample image

```bash
cd "$PROJECT_DIR"
mkdir -p data/sample_images
curl -fL --retry 3 \
  https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg \
  -o data/sample_images/truck.jpg
```

Setup is complete. For subsequent sessions, use [EPITO.md](EPITO.md).
For partition generation, use [SAM_Exps.md](SAM_Exps.md).
