# GPU session in one command

Before any experiment, run **one command on the login node**:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
bash src/hpc/gpu.sh v100          # or: h200, h100, a100, l40s, l4, t4, best
```

It asks Slurm for that GPU and waits for it. On the node it opens a shell that has already:
- activated the right Python environment for the node's CPU (ARM or x86);
- set `PROJECT_DIR` and `SAM2_DIR`, and changed to the repository;
- checked torch, the GPU, the BPT extension and the SAM packages.

When the prompt shows `GPU session ready on <node>`, run experiments directly, for example:

```bash
python src/scripts/generate_coco_partitions.py --config MSCOCO_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda --generate_at_least_n_masks 200 --save-partitions sam compact --compact-strategy greedy split
```

`exit` ends the session and frees the GPU.

## GPU names

| Name | GPU | Partition / GRES | CPU | Environment it activates |
| --- | --- | --- | --- | --- |
| `h200` | H200, 141 GB (heaviest) | `llm`, `gpu:h200` | x86_64 | `pt_312_sam_x86` |
| `h100` (or `gh200`) | GH200 Grace Hopper (H100) | `gracehopper`, `gpu:gh200` | aarch64 | `pt_312_sam` |
| `a100` | A100 (EPITO) | `epito`, `gpu:a100` | aarch64 | `pt_312_sam` |
| `l40s` | L40S, 48 GB | `cascadelake`, `gpu:l40` | x86_64 | `pt_312_sam_x86` |
| `v100` | V100S, 32 GB | `cascadelake`, `gpu:v100` | x86_64 | `pt_312_sam_x86` |
| `l4` | L4, 24 GB | `mirri`, `gpu:l4` | x86_64 | `pt_312_sam_x86` |
| `t4` | T4, 16 GB | `cascadelake`, `gpu:t4` | x86_64 | `pt_312_sam_x86` |
| `best` | heaviest GPU free right now, from `gpu_check.sh` | | | |

The environment is chosen on the node from its CPU type, so the GPU name only decides which node you get. The first time on an x86 node, the x86 environment is built automatically (10–20 minutes; see [OTHER_GPU_NODES.md](OTHER_GPU_NODES.md), Section 7).

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--time HH:MM:SS` | `06:00:00` | Session length (6 h is the partition limit) |
| `--node NAME` | any | One specific node, e.g. `--node cascadelake-V100S-02` |
| `--gpus N` | 1 | GPUs per session (`h100` nodes have only 1) |
| `--cpus N`, `--mem SIZE` | 16, `64G` | CPU cores and memory |
| `--sam` | off | The check also loads SAM-1 ViT-B and SAM-2.1 Large on the GPU |
| `--tmux NAME` | off | Run the whole thing in a new tmux session on the login node |
| `-- COMMAND ...` | | Run one command instead of opening a shell, then free the GPU |

Examples:

```bash
bash src/hpc/gpu.sh a100 --time 02:00:00             # EPITO A100 for 2 hours
bash src/hpc/gpu.sh v100 --node cascadelake-V100S-02  # this V100S node
bash src/hpc/gpu.sh best --tmux exp1                  # heaviest free GPU, inside tmux
bash src/hpc/gpu.sh v100 -- python src/scripts/generate_busi_partitions.py --config BUSI_epito \
  --model medsam --device cuda --generate_at_least_n_masks 50 --save-partitions sam compact --compact-strategy greedy split --limit 1000
```

## Keep the session alive: `--tmux`

Without tmux, the session ends if your SSH connection or laptop drops. With `--tmux NAME`, it runs inside tmux on the login node:
- **Detach:** `Ctrl-b`, then `d`. The run keeps going after you log out.
- **Come back:** `tmux attach -t NAME`.
- **List sessions:** `tmux ls`.

## Which GPU is free?

```bash
bash src/hpc/gpu_check.sh --no-small      # every GPU node, heaviest first, with free GPUs and queue length
squeue -u $USER                           # your own sessions and jobs
```

A request for a busy GPU waits in the queue; Ctrl-C cancels it. Look at `squeue -u $USER` now and then: old sessions you forgot to `exit` keep holding a GPU. Cancel them with `scancel JOBID`.

## Good to know

- **6-hour limit.** The session ends after `--time`. Long runs resume per image, so start a new session and rerun the same command.
- **Results are the same on every GPU**, up to floating-point differences, but timings are not. Note the GPU when reporting times; the node name is in the prompt and in the job log.
- **Batch jobs instead of a shell:** see [OTHER_GPU_NODES.md](OTHER_GPU_NODES.md), Section 0 (`gpu_check.sh --sbatch` with the chained `.sbatch` files).
- **Already on a node** (e.g. after a plain `srun`): `source src/hpc/node_env.sh` does the same setup in the current shell.

## How it works

`src/hpc/gpu.sh` maps the GPU name to a Slurm partition and GRES, then runs:

```bash
srun -p <partition> --gres=gpu:<type>:1 --cpus-per-task=16 --mem=64G --time=06:00:00 --pty bash --rcfile src/hpc/gpu_rc.sh -i
```

`src/hpc/gpu_rc.sh` is the shell's startup file. It loads `~/.bashrc` and then sources `src/hpc/node_env.sh`, which detects the CPU and GPU, picks and if needed builds the environment, and runs the check. With `-- COMMAND`, the same setup runs before the command in a non-interactive shell.
