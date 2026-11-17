#!/usr/bin/env bash
#SBATCH --job-name=XAI-budgets
#SBATCH --partition=epito
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --chdir=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
#SBATCH --output=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/XAI-budgets_%j.out
#SBATCH --error=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/XAI-budgets_%j.err

# Usage (from the repository root, after mkdir -p logs):
#   sbatch --export=ALL,CHAIN_LEFT=40 src/hpc/run_xai_budgets.sbatch --budgets 3000 5000 -- \
#       src/scripts/explanations/OD_yolo.py --config MSCOCO_epito_sam1 --show-gt --explain_all --no-heatmaps
# Each job first queues its follow-up (runs after this one ends, CHAIN_LEFT more times at most).
# The driver cancels the follow-up when every budget is done or a budget keeps failing.
# --signal sends TERM two minutes before the time limit, so the attempt is recorded cleanly.
set -euo pipefail
if (( $# < 1 )); then
    echo 'Usage: sbatch [--export=ALL,CHAIN_LEFT=N] src/hpc/run_xai_budgets.sbatch --budgets B... -- RUNNER.py [args...]' >&2
    exit 2
fi
CHAIN_LEFT=${CHAIN_LEFT:-0}
if (( CHAIN_LEFT > 0 )); then
    CHAIN_NEXT_JOB=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,CHAIN_LEFT=$((CHAIN_LEFT - 1)) src/hpc/run_xai_budgets.sbatch "$@")
    export CHAIN_NEXT_JOB
    echo "Follow-up job $CHAIN_NEXT_JOB queued ($((CHAIN_LEFT - 1)) more allowed after it)"
fi
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
source "${SAM_ENV:-/beegfs/home/mrashid/pt_312_sam}/bin/activate"  # Override with --export=ALL,SAM_ENV=... on other nodes
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
exec python src/scripts/explanations/run_budgets.py "$@"
