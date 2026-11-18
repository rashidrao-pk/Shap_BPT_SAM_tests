#!/usr/bin/env bash
# Set up and activate the right Python environment on whatever node this shell is on.
# Run inside the node's shell (after srun ... --pty bash), from the repository root:
#
#   source src/hpc/node_env.sh            # detect hardware, build the environment if missing, activate, check
#   source src/hpc/node_env.sh --setup    # also rerun the installation (x86_64 nodes), e.g. after new requirements
#   source src/hpc/node_env.sh --sam      # also load SAM-1 ViT-B and SAM-2.1 Large on the GPU
#   bash src/hpc/node_env.sh              # same checks, but the activation does not stay in your shell
#
# aarch64 (epito, gracehopper) -> /beegfs/home/mrashid/pt_312_sam   (pt_312_sam_gh instead, if it exists, on gh nodes)
# x86_64  (llm H200, L40S, V100S, T4, L4, login node) -> /beegfs/home/mrashid/pt_312_sam_x86, built by setup_x86_env.sh
# Set SAM_ENV beforehand to force another environment.

_node_env_main() {
  local setup=0 sam=0 arg
  for arg in "$@"; do
    case $arg in
      --setup) setup=1 ;;
      --sam) sam=1 ;;
      -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; return 0 ;;
      *) echo "Unknown option $arg" >&2; return 2 ;;
    esac
  done

  export PROJECT_DIR=${PROJECT_DIR:-/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests}
  export SAM2_DIR=${SAM2_DIR:-/beegfs/home/mrashid/repos/third_party/sam2}
  local arch host gpu env
  arch=$(uname -m)
  host=$(hostname -s)
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c | sed 's/^ *//' | paste -sd, -)

  if [[ -n ${SAM_ENV:-} ]]; then
    env=$SAM_ENV
  elif [[ $arch == aarch64 ]]; then
    env=/beegfs/home/mrashid/pt_312_sam
    [[ $host == gh* && -x /beegfs/home/mrashid/pt_312_sam_gh/bin/python ]] && env=/beegfs/home/mrashid/pt_312_sam_gh
  elif [[ $arch == x86_64 ]]; then
    env=/beegfs/home/mrashid/pt_312_sam_x86
  else
    echo "Unknown CPU architecture $arch" >&2; return 1
  fi
  echo "Node $host | CPU $arch | GPU ${gpu:-none} | environment $env"
  [[ -z $gpu ]] && echo 'No GPU on this node: fine for viewing and evaluation, not for SAM or XAI runs.'

  deactivate 2>/dev/null || true
  unset PYTHONPATH

  if [[ ! -x $env/bin/python || $setup == 1 ]]; then
    if [[ $arch == x86_64 ]]; then
      [[ -z $gpu ]] && { echo "Build $env on a GPU node (the installation check needs CUDA)." >&2; return 1; }
      echo "Building $env with src/hpc/setup_x86_env.sh (about 10-20 minutes the first time) ..."
      SAM_ENV=$env PROJECT_DIR=$PROJECT_DIR SAM2_DIR=$SAM2_DIR bash "$PROJECT_DIR/src/hpc/setup_x86_env.sh" || return 1
    elif [[ ! -x $env/bin/python ]]; then
      echo "$env is missing. Build it as in docs/Partitions/full/EPITO_SETUP.md." >&2; return 1
    else
      echo '--setup only rebuilds the x86_64 environment; on aarch64 the BPT extension is rebuilt below if needed.'
    fi
  fi

  # shellcheck disable=SC1091
  source "$env/bin/activate"
  cd "$PROJECT_DIR" || return 1
  echo "Activated $(command -v python)"

  echo 'Importing torch (the first import on a node can take a minute) ...'
  if ! python -c 'import torch' 2>/dev/null; then
    echo "torch does not import with $env on $host." >&2
    python -c 'import torch' 2>&1 | tail -3 >&2
    return 1
  fi

  echo 'Checking the BPT extension ...'
  if ! python -c 'from shap_bpt import bpt' 2>/dev/null; then
    echo 'BPT extension missing for this CPU type; compiling it ...'
    python setup.py build_ext --inplace >/dev/null || { echo 'Compiling the BPT extension failed' >&2; return 1; }
  fi

  echo 'Checking GPU and SAM packages ...'
  SAM_CHECK=$sam NODE_GPU=$gpu python - <<'PY' || { echo "The check failed; see the message above. Try: source src/hpc/node_env.sh --setup" >&2; return 1; }
import os, numpy, torch
from shap_bpt import bpt
print("numpy", numpy.__version__, "| torch", torch.__version__, "| CUDA", torch.version.cuda, "|", os.path.dirname(torch.__file__))
if torch.cuda.is_available():
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("GPU:", ", ".join(names), "| capability", torch.cuda.get_device_capability(0))
    print("CUDA calculation:", (torch.ones(4, device="cuda") + 1).cpu().tolist())
else:
    print("CUDA not available (CPU only)")
    if os.environ.get("NODE_GPU"):
        raise SystemExit(f"This node has a GPU ({os.environ['NODE_GPU']}) but torch {torch.__version__} cannot use it.")
print("BPT extension:", os.path.basename(bpt.__file__))
import segment_anything, sam2
print("SAM-1 and SAM-2 packages import OK")
if os.environ.get("SAM_CHECK") == "1" and torch.cuda.is_available():
    from sam2.build_sam import build_sam2
    from segment_anything import sam_model_registry
    build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
               os.path.join(os.environ["SAM2_DIR"], "checkpoints/sam2.1_hiera_large.pt"), device="cuda")
    print("SAM-2.1 Large loads on the GPU")
    sam_model_registry["vit_b"](checkpoint="checkpoints/sam1/sam_vit_b_01ec64.pth").to("cuda")
    print("SAM-1 ViT-B loads on the GPU")
PY
  echo "Ready on $host. For sbatch jobs on this hardware: --export=ALL,SAM_ENV=$env"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  _node_env_main "$@" || exit 1
  echo '(Run with "source" to keep the environment active in your shell.)'
else
  _node_env_main "$@"
fi
