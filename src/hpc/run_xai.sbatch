#!/usr/bin/env bash
#SBATCH --job-name=XAI
#SBATCH --partition=epito
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
#SBATCH --output=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/XAI_%j.out
#SBATCH --error=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/XAI_%j.err

set -euo pipefail
# Usage: mkdir -p logs; sbatch src/hpc/run_xai.sbatch SCRIPT.py [script arguments...]
# Re-submit the same command after timeout. Completed-image resume belongs to the runner.
if (( $# < 1 )); then
    echo 'Usage: sbatch src/hpc/run_xai.sbatch SCRIPT.py [arguments...]' >&2
    exit 2
fi
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
exec python "$@"
