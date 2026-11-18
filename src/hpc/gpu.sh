#!/usr/bin/env bash
# One command from the login node: get a GPU of the named type and open a shell there with the right Python
# environment already active and checked (src/hpc/node_env.sh). See docs/Partitions/full/GPU.md.
#
#   bash src/hpc/gpu.sh v100                     # V100S, 6 h, interactive shell ready for experiments
#   bash src/hpc/gpu.sh a100 --time 02:00:00     # EPITO A100 for 2 h
#   bash src/hpc/gpu.sh h100 --tmux h100         # inside tmux, so the session survives an SSH drop
#   bash src/hpc/gpu.sh best                     # heaviest GPU free right now (gpu_check.sh)
#   bash src/hpc/gpu.sh v100 -- python src/scripts/generate_coco_partitions.py ...   # run one command, no shell
#
# GPU names: h200 | h100 (= gh200) | a100 | l40s | v100 | l4 | t4 | best
# Options:   --time HH:MM:SS (default 06:00:00, the partition limit)   --node NODE (one specific node)
#            --cpus N (16)   --mem SIZE (64G)   --gpus N (1)   --sam (also load SAM-1/SAM-2 in the check)
#            --tmux NAME (run the whole thing in a new tmux session on the login node)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}"; }

gpu_options() {  # name -> "partition gres-type label"
    case $(tr "[:upper:]" "[:lower:]" <<<"$1") in
        h200) echo "llm h200 H200" ;;
        h100|gh200|gh) echo "gracehopper gh200 GH200(H100)" ;;
        a100|epito) echo "epito a100 A100" ;;
        l40s|l40) echo "cascadelake l40 L40S" ;;
        v100|v100s) echo "cascadelake v100 V100S" ;;
        l4) echo "mirri l4 L4" ;;
        t4) echo "cascadelake t4 T4" ;;
        *) return 1 ;;
    esac
}

main() {
    (($#)) || { usage; return 2; }
    local gpu=$1; shift
    [[ $gpu == -h || $gpu == --help ]] && { usage; return 0; }
    local time=06:00:00 node='' cpus=16 mem=64G count=1 sam=0 tmux_name='' command=()
    local original=("$gpu" "$@")
    while (($#)); do
        case $1 in
            --time) time=$2; shift ;;
            --node) node=$2; shift ;;
            --cpus) cpus=$2; shift ;;
            --mem) mem=$2; shift ;;
            --gpus) count=$2; shift ;;
            --sam) sam=1 ;;
            --tmux) tmux_name=$2; shift ;;
            --) shift; command=("$@"); break ;;
            *) echo "Unknown option $1" >&2; usage; return 2 ;;
        esac
        shift
    done

    if [[ -n $tmux_name ]]; then  # Re-run this same command inside tmux, without --tmux.
        command -v tmux >/dev/null || { echo 'tmux not found' >&2; return 1; }
        local rest=() skip=0 arg
        for arg in "${original[@]}"; do
            if ((skip)); then skip=0; continue; fi
            [[ $arg == --tmux ]] && { skip=1; continue; }
            rest+=("$arg")
        done
        echo "Starting tmux session '$tmux_name' (detach: Ctrl-b d; back: tmux attach -t $tmux_name)"
        exec tmux new-session -s "$tmux_name" "bash $(printf '%q ' "${BASH_SOURCE[0]}" "${rest[@]}"); exec bash"
    fi

    command -v srun >/dev/null || { echo 'srun not found: run this on the mercurio login node' >&2; return 1; }
    local partition type label
    if [[ $(tr "[:upper:]" "[:lower:]" <<<"$gpu") == best ]]; then
        eval "$(bash "$ROOT/src/hpc/gpu_check.sh" --sbatch)"   # sets GPU_OPTS, GPU_NODE, SAM_ENV
        read -r _ partition _ <<<"$GPU_OPTS"
        type=$(sed -E 's/.*--gres=gpu:([^:]+):.*/\1/' <<<"$GPU_OPTS")
        label=$type
        [[ -z $node && -n ${GPU_NODE:-} ]] && node=$GPU_NODE
    else
        local spec
        spec=$(gpu_options "$gpu") || {
            echo "Unknown GPU '$gpu'. Use: h200 h100 a100 l40s v100 l4 t4 best" >&2; return 2; }
        read -r partition type label <<<"$spec"
    fi
    local opts=(-p "$partition" --gres="gpu:$type:$count" --cpus-per-task="$cpus" --mem="$mem" --time="$time"
                -J "ShapBPT $label")
    [[ -n $node ]] && opts+=(--nodelist="$node")

    echo "GPU: $label x $count on partition $partition${node:+ (node $node)}, $time"
    sinfo -h -p "$partition" -N -O "NodeHost:26,StateCompact:10,Gres:30" 2>/dev/null \
        | grep -i "gpu:$type" | sed 's/^/  /' || true
    echo "Waiting for the allocation (Ctrl-C cancels) ..."

    export PROJECT_DIR=$ROOT NODE_ENV_ARGS=$([[ $sam == 1 ]] && echo --sam || true)
    if ((${#command[@]})); then
        # One command, then the allocation ends.
        exec srun "${opts[@]}" --pty bash -c \
            'source "$PROJECT_DIR/src/hpc/node_env.sh" $NODE_ENV_ARGS || exit 1; exec "$@"' bash "${command[@]}"
    fi
    # Interactive shell whose startup file sources ~/.bashrc and then node_env.sh.
    exec srun "${opts[@]}" --pty bash --rcfile "$ROOT/src/hpc/gpu_rc.sh" -i
}

main "$@"
