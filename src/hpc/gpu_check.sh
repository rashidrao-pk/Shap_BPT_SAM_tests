#!/usr/bin/env bash
# Which GPU can I get right now? Run on the mercurio login node, from the repository root.
# See docs/Partitions/full/OTHER_GPU_NODES.md.
#
#   bash src/hpc/gpu_check.sh                  # table of GPU nodes, heaviest first, with free GPUs
#   bash src/hpc/gpu_check.sh --start          # interactive session on the best free GPU
#   bash src/hpc/gpu_check.sh --start --time 02:00:00
#   eval "$(bash src/hpc/gpu_check.sh --sbatch)"   # sets GPU_OPTS, GPU_NODE and SAM_ENV for sbatch
#   bash src/hpc/gpu_check.sh --no-small       # leave out T4 and L4
#
# After eval: sbatch $GPU_OPTS --export=ALL,CHAIN_LEFT=10 src/hpc/run_partitions.sbatch ...
# (--export=ALL carries SAM_ENV into the job, so it activates the environment for that CPU type.)

# partition  GRES type  label           CPU      small
CANDIDATES=(
  "llm         h200   H200-141GB      x86_64   no"
  "gracehopper gh200  GH200(H100)     aarch64  no"
  "epito       a100   A100            aarch64  no"
  "cascadelake l40    L40S-48GB       x86_64   no"
  "cascadelake v100   V100S-32GB      x86_64   no"
  "mirri       l4     L4-24GB         x86_64   yes"
  "cascadelake t4     T4-16GB         x86_64   yes"
  "mirri       t4     T4-16GB         x86_64   yes"
)
ENV_ARM=${ENV_ARM:-/beegfs/home/mrashid/pt_312_sam}
ENV_X86=${ENV_X86:-/beegfs/home/mrashid/pt_312_sam_x86}
CPUS=${CPUS:-16}
MEM=${MEM:-64G}

# Reads `sinfo -N -O NodeHost,StateCompact,Gres,GresUsed` lines; prints "node state total used free"
# for the nodes carrying GPU type $1. Only idle and mixed nodes count as free.
parse_free() {
  awk -v type="$1" '
    function count(s,   m) {
      if (match(s, "gpu:" type ":[0-9]+")) { m = substr(s, RSTART, RLENGTH); sub(/.*:/, "", m); return m + 0 }
      return -1
    }
    {
      total = count($3); if (total < 0) next
      used = count($4); if (used < 0) used = 0
      free = ($2 ~ /^(idle|mix)-?$/) ? total - used : 0
      if (free < 0) free = 0
      print $1, $2, total, used, free
    }'
}

node_lines() {  # partition type
  sinfo -h -N -p "$1" -O "NodeHost:40,StateCompact:14,Gres:80,GresUsed:80" 2>/dev/null | parse_free "$2"
}

env_for() { [[ $1 == aarch64 ]] && echo "$ENV_ARM" || echo "$ENV_X86"; }

main() {
  local start=0 sbatch=0 small=1 time=06:00:00
  while (($#)); do
    case $1 in
      --start) start=1 ;;
      --sbatch) sbatch=1 ;;
      --no-small) small=0 ;;
      --time) time=$2; shift ;;
      -h|--help) sed -n '2,13p' "$0"; return 0 ;;
      *) echo "Unknown option $1" >&2; return 2 ;;
    esac
    shift
  done
  command -v sinfo >/dev/null || { echo 'sinfo not found: run this on the mercurio login node' >&2; return 1; }

  local best='' fallback='' line part type label arch is_small node state total used free nodes_up env
  ((sbatch)) || printf '%-12s %-6s %-13s %-8s %-24s %-9s %4s %5s %8s\n' \
    PARTITION GRES GPU CPU NODE STATE GPUS FREE QUEUED
  for line in "${CANDIDATES[@]}"; do
    read -r part type label arch is_small <<<"$line"
    [[ $is_small == yes && $small == 0 ]] && continue
    local queued
    queued=$(squeue -h -p "$part" -t PENDING 2>/dev/null | wc -l | tr -d ' ')
    nodes_up=0
    while read -r node state total used free; do
      [[ -z $node ]] && continue
      ((sbatch)) || printf '%-12s %-6s %-13s %-8s %-24s %-9s %4s %5s %8s\n' \
        "$part" "$type" "$label" "$arch" "$node" "$state" "$total" "$free" "$queued"
      [[ $state =~ ^(idle|mix|alloc)-?$ ]] && nodes_up=1
      if ((free > 0)) && [[ -z $best ]]; then best="$part $type $label $arch $node"; fi
    done < <(node_lines "$part" "$type")
    if ((nodes_up)) && [[ -z $fallback ]]; then fallback="$part $type $label $arch"; fi
  done

  local chosen=$best queue_note=''
  if [[ -z $chosen ]]; then
    chosen=$fallback
    queue_note='no GPU is free now; the request will queue'
  fi
  if [[ -z $chosen ]]; then echo 'No usable GPU node found (all down or drained).' >&2; return 1; fi
  read -r part type label arch node <<<"$chosen"
  env=$(env_for "$arch")
  local opts="-p $part --gres=gpu:$type:1 --cpus-per-task=$CPUS --mem=$MEM"

  if ((sbatch)); then
    [[ -n $queue_note ]] && echo "echo '$queue_note' >&2"
    printf 'GPU_OPTS=%q\nGPU_NODE=%q\nexport SAM_ENV=%q\n' "$opts" "${node:-}" "$env"
    echo "echo 'GPU: $label on ${node:-$part} ($arch); SAM_ENV=$env' >&2"
    return 0
  fi

  echo
  echo "Best choice: $label on ${node:-partition $part} ($arch)${queue_note:+ - $queue_note}"
  echo "Environment: $env"
  echo "Interactive: srun $opts --time=$time --pty bash"
  echo "  then:      cd $(pwd) && source src/hpc/node_env.sh"
  echo "Batch jobs:  eval \"\$(bash src/hpc/gpu_check.sh --sbatch)\"; sbatch \$GPU_OPTS --export=ALL,CHAIN_LEFT=10 <file.sbatch> ..."
  if ((start)); then
    echo
    echo "Starting the session. Wait for the prompt on the node, then: source src/hpc/node_env.sh"
    exec srun $opts -J "ShapBPT $label" --time="$time" --pty bash
  fi
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
