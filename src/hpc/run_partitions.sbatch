#!/usr/bin/env bash
#SBATCH --job-name=partitions
#SBATCH --partition=epito
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
#SBATCH --output=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/partitions_%j.out
#SBATCH --error=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/partitions_%j.err

# Usage (from the repository root, after mkdir -p logs):
#   sbatch --export=ALL,CHAIN_LEFT=10 src/hpc/run_partitions.sbatch src/scripts/generate_coco_partitions.py \
#       --config MSCOCO_epito --model sam1 --checkpoint ... --max-regions 126 --save-partitions sam compact \
#       --generate_at_least_n_masks 200 --device cuda
# The generators resume per image. Each job queues its follow-up first (at most CHAIN_LEFT more); when the
# generator finishes every image (exit 0), the follow-up is cancelled. A time-limit kill leaves it queued.
set -uo pipefail
if (( $# < 1 )); then
    echo 'Usage: sbatch [--export=ALL,CHAIN_LEFT=N] src/hpc/run_partitions.sbatch GENERATOR.py [args...]' >&2
    exit 2
fi
CHAIN_LEFT=${CHAIN_LEFT:-0}
NEXT=''
if (( CHAIN_LEFT > 0 )); then
    NEXT=$(sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" \
        --export=ALL,CHAIN_LEFT=$((CHAIN_LEFT - 1)) src/hpc/run_partitions.sbatch "$@")
    echo "Follow-up job $NEXT queued ($((CHAIN_LEFT - 1)) more allowed after it)"
fi
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python "$@"
status=$?
if (( status == 0 )) && [[ -n "$NEXT" ]]; then
    scancel "$NEXT" && echo "All images done: cancelled follow-up job $NEXT"
fi
exit $status
