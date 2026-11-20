#!/usr/bin/env bash
# Epito (A100) helper for Shap_BPT_SAM_tests on HPC4AI / Mercurio.
# Always run with Bash explicitly.
#
# Modes
#   bash src/hpc/setup_epito.sh
#       As before: interactive srun on the open `epito` partition (6 h max),
#       then a tmux session on the compute node with the environment ready.
#
#   RES=<ReservationName> [GPUS=1|2] bash src/hpc/setup_epito.sh hold [HH:MM:SS]
#       Submit a placeholder job (sleep infinity) inside your Booked reservation
#       on `epito-booked`. Prints the JOBID. Default time 20:00:00; keep it
#       within the reservation window (check: scontrol show res <name>).
#
#   bash src/hpc/setup_epito.sh attach <JOBID>
#       Open an interactive shell (env ready) inside that running job.
#       Run this from a tmux session on the LOGIN node so it survives disconnects.
#       You can attach as many shells as you like to the same job.
#
#   bash src/hpc/setup_epito.sh status
#       Show your jobs and your reservations.
#
# When finished with a held job:  scancel <JOBID>

if [[ $- == *i* ]]; then
    # This file also serves as the interactive shell's rcfile. Skip ~/.bashrc,
    # whose micromamba initialization may use a binary for another architecture.
    if [[ ! -f /beegfs/home/mrashid/pt_312_sam/bin/activate ]]; then
        printf 'Python environment is missing: /beegfs/home/mrashid/pt_312_sam\n' >&2
        return 1
    fi

    source /beegfs/home/mrashid/pt_312_sam/bin/activate
    export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/
    export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
    export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
    cd "$PROJECT_DIR"
    if ! python --version; then
        printf 'Python could not run. Check that pt_312 supports this compute node architecture.\n' >&2
        return 1
    fi
    printf 'Environment ready in %s (job %s on %s).\n' "$PWD" "${SLURM_JOB_ID:-none}" "$(hostname)"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    return
fi

set -euo pipefail
script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/setup_epito.sh"
project_dir="$(dirname -- "$(dirname -- "$script_path")")"
job_name=ShapBPT

case ${1:-} in
    hold)
        : "${RES:?set RES=<ReservationName> (see: scontrol show res)}"
        if ! scontrol show res "$RES" >/dev/null 2>&1; then
            printf 'Reservation "%s" not found. Your reservations:\n' "$RES" >&2
            scontrol show res | grep -B1 -A3 "Users=.*\b$USER\b" >&2 || printf '  (none)\n' >&2
            exit 1
        fi
        time_limit=${2:-20:00:00}
        jobid=$(sbatch --parsable \
            -p epito-booked --reservation="$RES" \
            --gres=gpu:a100:"${GPUS:-1}" \
            -J "$job_name" --time="$time_limit" \
            -o "$HOME/holder_%j.log" \
            --wrap 'sleep infinity')
        printf 'Submitted holder job %s in reservation %s (time %s).\n' "$jobid" "$RES" "$time_limit"
        printf 'Wait for state R:   squeue -j %s\n' "$jobid"
        printf 'Then (inside login-node tmux):   bash %s attach %s\n' "$script_path" "$jobid"
        printf 'When done:          scancel %s\n' "$jobid"
        exit 0
        ;;
    attach)
        jobid=${2:?usage: bash setup_epito.sh attach <JOBID>}
        state=$(squeue -h -j "$jobid" -o %T 2>/dev/null || true)
        if [[ $state != RUNNING ]]; then
            printf 'Job %s is not running (state: %s). Check: squeue --me\n' "$jobid" "${state:-not found}" >&2
            exit 1
        fi
        if [[ -z ${TMUX:-} ]]; then
            printf 'Tip: run this inside tmux on the login node so the shell survives disconnects.\n' >&2
        fi
        exec srun --jobid="$jobid" --overlap --pty \
            bash --noprofile --rcfile "$script_path" -i
        ;;
    status)
        squeue --me -o '%.10i %.14P %.10j %.8T %.10M %.10l %R'
        printf '\nYour reservations:\n'
        scontrol show res | grep -B1 -A3 "Users=.*\b$USER\b" || printf '  (none)\n'
        exit 0
        ;;
esac

# ---- Default mode: open `epito` partition, 6 h max (original behaviour) ----
if [[ ${1:-} != --compute && -z ${SLURM_JOB_ID:-} ]]; then
    exec srun -p epito --gres=gpu:a100:1 -J "$job_name" --pty \
        bash --noprofile --norc "$script_path" --compute
fi

cd -- "$project_dir"
if [[ -n ${TMUX:-} ]]; then
    exec bash --noprofile --rcfile "$script_path" -i
fi

# Quote the rcfile path for the shell tmux uses to launch the command.
printf -v shell_command 'exec bash --noprofile --rcfile %q -i' "$script_path"
exec tmux new-session -s YOLO -c "$project_dir" "$shell_command"
