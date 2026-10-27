## INSTALL SAM3

```bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH

cd /beegfs/home/mrashid/repos/third_party

git clone https://github.com/facebookresearch/sam3.git
cd sam3


python -m pip install \
  "einops>=0.8.0" \
  "timm>=1.0.17" \
  "ftfy==6.1.1" \
  regex \
  huggingface_hub \
  --no-deps

python -m pip install -e . \
  --no-deps \
  --no-build-isolation

```

### Step 1 — Authenticate with Hugging Face

```bash
hf auth login
hf auth whoami
```

Then verify repository access:

```bash
python - <<'PY'
from huggingface_hub import list_repo_files

files = list_repo_files("facebook/sam3")

print("SAM3 repository access: SUCCESS")
for name in files:
    if name.endswith((".pt", ".pth", ".safetensors")):
        print("Checkpoint:", name)
PY
```

Step 2 — Test SAM3 inference on the truck image

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
