#!/usr/bin/env bash
# Build the x86_64 environment for the H200 (llm) and L40S nodes; see docs/Partitions/full/OTHER_GPU_NODES.md.
# Run inside a session on the node, from the repository root:
#   bash src/hpc/setup_x86_env.sh
# Safe to rerun: an existing environment is reused and packages already installed are skipped by pip.
# The environment has its own PyTorch (official CUDA 12.6 wheels, same version as EPITO): the node driver
# supports CUDA <= 12.9, and pinning stops other packages from pulling in a newer torch (e.g. 2.15+cu132).
set -euo pipefail
deactivate 2>/dev/null || true
unset PYTHONPATH

PROJECT_DIR=${PROJECT_DIR:-/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests}
SAM2_DIR=${SAM2_DIR:-/beegfs/home/mrashid/repos/third_party/sam2}
SAM_ENV=${SAM_ENV:-/beegfs/home/mrashid/pt_312_sam_x86}
TORCH_VERSION=${TORCH_VERSION:-2.7.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.22.1}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

echo "Host $(hostname), $(uname -m); environment $SAM_ENV"
[[ $(uname -m) == x86_64 ]] || { echo 'This script is for x86_64 nodes; EPITO and gracehopper use pt_312_sam' >&2; exit 1; }
[[ -d $SAM2_DIR ]] || { echo "No SAM-2 checkout at $SAM2_DIR; set SAM2_DIR" >&2; exit 1; }

[[ -x $SAM_ENV/bin/python ]] || python3.12 -m venv "$SAM_ENV"
# shellcheck disable=SC1091
source "$SAM_ENV/bin/activate"
SITE="$SAM_ENV/lib/python3.12/site-packages"
rm -f "$SITE/opt_pytorch.pth"   # Older versions of this script borrowed /opt PyTorch; no longer used

python -m pip install --upgrade pip setuptools wheel Cython
want="$TORCH_VERSION+${TORCH_INDEX##*/}"
have=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
if [[ $have != "$want" ]]; then
  echo "Replacing torch '${have:-none}' with $want"
  python -m pip uninstall -y torch torchvision torchaudio triton || true
  python -m pip freeze | grep -E '^nvidia-' | cut -d= -f1 | xargs -r python -m pip uninstall -y
  python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX"
fi
# Every later install must keep this torch.
CONSTRAINTS="$SAM_ENV/torch-constraints.txt"
python -m pip freeze | grep -E '^(torch|torchvision|triton|nvidia-[^=]+)==' > "$CONSTRAINTS"
PIP=(python -m pip install -c "$CONSTRAINTS")

"${PIP[@]}" -r "$PROJECT_DIR/requirements-xai-all.txt"
"${PIP[@]}" PyYAML pycocotools scipy pandas pillow tabulate matplotlib transformers timm
"${PIP[@]}" --no-deps ultralytics
"${PIP[@]}" --no-deps git+https://github.com/facebookresearch/segment-anything.git
"${PIP[@]}" hydra-core iopath tqdm
SAM2_BUILD_CUDA=0 "${PIP[@]}" --no-deps -e "$SAM2_DIR"
cd "$PROJECT_DIR"
"${PIP[@]}" --no-deps -e .   # Builds bpt.cpython-312-x86_64-linux-gnu.so next to EPITO's aarch64 file

python - <<'PY'
import numpy, torch, shap_bpt
from shap_bpt import bpt
print("numpy", numpy.__version__, "| torch", torch.__version__, torch.__file__)
assert torch.cuda.is_available(), "torch cannot use the GPU; see the warning above"
print("GPU:", torch.cuda.get_device_name(0), "x", torch.cuda.device_count())
print("CUDA calculation:", (torch.ones(4, device="cuda") + 1).cpu())
print("BPT extension:", bpt.__file__)
import segment_anything, sam2
print("SAM-1 and SAM-2 import OK")
PY
echo "Done. Later sessions: source $SAM_ENV/bin/activate"
