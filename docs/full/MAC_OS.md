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

## Next sessions

```bash
conda activate shapbpt-sam
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd ~/repos/Shap_BPT_SAM_tests
```

Use [SAM_Exps.md](SAM_Exps.md) for partition generation. Set `PROJECT_DIR` and
`SAM2_DIR` to your actual checkout paths and use `--device auto` in place of
`--device cuda` in its commands.
