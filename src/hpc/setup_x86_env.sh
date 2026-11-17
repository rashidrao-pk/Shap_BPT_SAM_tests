#!/usr/bin/env bash
# Build the x86_64 environment for the H200 (llm) and L40S nodes; see docs/Partitions/full/OTHER_GPU_NODES.md.
# Run inside a session on the node, from the repository root:
#   bash src/hpc/setup_x86_env.sh
# Safe to rerun: an existing environment is reused and packages already installed are skipped by pip.
set -euo pipefail
deactivate 2>/dev/null || true
unset PYTHONPATH

PROJECT_DIR=${PROJECT_DIR:-/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests}
SAM2_DIR=${SAM2_DIR:-/beegfs/home/mrashid/repos/third_party/sam2}
SAM_ENV=${SAM_ENV:-/beegfs/home/mrashid/pt_312_sam_x86}
OPT_TORCH=${OPT_TORCH:-/opt/pytorch-v2.7.1/lib/python3.12/site-packages}

echo "Host $(hostname), $(uname -m); environment $SAM_ENV"
[[ $(uname -m) == x86_64 ]] || { echo 'This script is for x86_64 nodes; EPITO and gracehopper use pt_312_sam' >&2; exit 1; }
[[ -d $OPT_TORCH/torch ]] || { echo "No PyTorch at $OPT_TORCH on this node; set OPT_TORCH" >&2; exit 1; }
[[ -d $SAM2_DIR ]] || { echo "No SAM-2 checkout at $SAM2_DIR; set SAM2_DIR" >&2; exit 1; }

[[ -x $SAM_ENV/bin/python ]] || python3.12 -m venv "$SAM_ENV"
# shellcheck disable=SC1091
source "$SAM_ENV/bin/activate"
# The node's PyTorch stays visible even when PYTHONPATH is unset, as in the sbatch files.
echo "$OPT_TORCH" > "$SAM_ENV/lib/python3.12/site-packages/opt_pytorch.pth"

python -m pip install --upgrade pip setuptools wheel Cython
# Never pip-install torch or torchvision here: they come from /opt and match the node's driver.
python -m pip install -r "$PROJECT_DIR/requirements-xai-all.txt"
python -m pip install PyYAML pycocotools scipy pandas pillow tabulate matplotlib transformers timm
python -m pip install --no-deps ultralytics
python -m pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
python -m pip install hydra-core iopath tqdm
SAM2_BUILD_CUDA=0 python -m pip install --no-deps -e "$SAM2_DIR"
cd "$PROJECT_DIR"
python -m pip install --no-deps -e .   # Builds bpt.cpython-312-x86_64-linux-gnu.so next to EPITO's aarch64 file

python - <<'PY'
import numpy, torch, shap_bpt
from shap_bpt import bpt
print("numpy", numpy.__version__, "| torch", torch.__version__, torch.__file__)
print("GPU:", torch.cuda.get_device_name(0), "x", torch.cuda.device_count())
print("CUDA calculation:", (torch.ones(4, device="cuda") + 1).cpu())
print("BPT extension:", bpt.__file__)
import segment_anything, sam2
print("SAM-1 and SAM-2 import OK")
PY
echo "Done. Later sessions: source $SAM_ENV/bin/activate"
