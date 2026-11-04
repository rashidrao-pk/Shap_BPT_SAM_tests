# EPITO: one-time SAM3 setup

Optional SAM3 installation notes, separate from the SAM1/SAM2 setup.
These notes use a different package configuration (`unset PYTHONPATH`).
After installation, see [SAM_Exps.md](SAM_Exps.md#optional-sam3-example).

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


Installation and authentication are one-time steps. Reauthenticate only if access expires.
