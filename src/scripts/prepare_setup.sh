#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="/beegfs/home/mrashid/repos/XAI/shap_bpt_sam"
VENV_ACTIVATE="/beegfs/home/mrashid/pt_312/bin/activate"
PYTORCH_PACKAGES="/opt/pytorch-v2.7.1/lib/python3.12/site-packages"

srun \
    --partition=epito \
    --gres=gpu:a100:1 \
    --pty \
    bash -c "
        source '${VENV_ACTIVATE}'
        export PYTHONPATH='${PYTORCH_PACKAGES}':\"\${PYTHONPATH:-}\"
        cd '${PROJECT_DIR}'
        echo 'Setup ready on host:' \"\$(hostname)\"
        exec bash -i
    "