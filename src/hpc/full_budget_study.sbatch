#!/usr/bin/env bash
#SBATCH --job-name=Full-XAI-budget
#SBATCH --partition=epito
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
#SBATCH --output=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/full_budget_%j.out
#SBATCH --error=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/full_budget_%j.err
set -euo pipefail
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
STUDY_HOME=$1
STUDY_ALLOCATION=${STUDY_ALLOCATION:-1}
STUDY_MAX_ALLOCATIONS=${STUDY_MAX_ALLOCATIONS:-40}
continuation_queued=0

queue_continuation() {
    if (( continuation_queued == 0 && STUDY_ALLOCATION < STUDY_MAX_ALLOCATIONS )); then
        # Queue only this frozen study; dependency prevents overlapping controllers.
        if sbatch --dependency="afterany:$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
            --export="ALL,STUDY_ALLOCATION=$((STUDY_ALLOCATION+1))" \
            "$STUDY_HOME/study.sbatch" "$STUDY_HOME"; then
            continuation_queued=1
        fi
    fi
}
trap queue_continuation USR1
python "$STUDY_HOME/driver.py" "$STUDY_HOME/manifest.json" &
controller_pid=$!
# A scheduler warning interrupts wait; keep waiting for the existing controller.
while true; do
    if wait "$controller_pid"; then
        exit 0
    else
        result=$?
        if (( result > 128 )) && kill -0 "$controller_pid" 2>/dev/null; then
            continue
        fi
        exit "$result"
    fi
done
