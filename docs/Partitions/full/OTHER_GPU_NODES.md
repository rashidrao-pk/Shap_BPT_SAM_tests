# Running on another GPU node when EPITO is busy

The HPC4AI (mercurio) cluster has other GPU nodes. Two are stronger than EPITO's A100s:
- the **`llm-h200` node with two NVIDIA H200**, the heaviest GPUs on the cluster;
- the **Grace Hopper nodes (`gh01`–`gh04`, partition `gracehopper`) with one GH200 (H100) each**. They are the easiest to use, because they are ARM like EPITO and run the EPITO environment unchanged.

This guide sets up both and runs the usual commands there. [EPITO_SETUP.md](EPITO_SETUP.md) stays the reference for the full installation.

## Which node

| Node type | GPUs | CPU architecture | Use |
| --- | --- | --- | --- |
| `epito` (4 nodes) | 2 × A100 | aarch64 (Arm Neoverse N1) | Default |
| **`llm-h200`** (partition `llm`, GRES `gpu:h200:2`) | **2 × H200** (141 GB each) | x86_64 | **Heaviest GPU**; needs its own environment (Section 7) |
| **`gh01`–`gh04`** (partition `gracehopper`, GRES `gpu:gh200:1`) | **1 × GH200** (H100) | **aarch64** (Grace) | **Easiest when EPITO is busy**: the EPITO environment runs as it is |
| `cascadelake-L40S` (partition `cascadelake`, GRES `gpu:l40:2`) | 2 × L40S | x86_64 | Needs its own environment (Section 6) |
| `cascadelake-V100S-*` (partition `cascadelake`, GRES `gpu:v100:2`) | 2 × V100S | x86_64 | Older; only if everything above is busy |
| `cascadelake-T4`, `mirri-T4`, `mirri-L4` | T4, L4 | x86_64 | Too small for SAM-2 Large at speed |

Facts that make the `gh` nodes easy to use:
- **Same storage:** `/beegfs/home/mrashid` is the same on every node. The repository, datasets, checkpoints, partition folders, results and the `*_epito.yaml` configs work unchanged.
- **Same CPU family:** `gh` is aarch64 like EPITO. The Python environment and the compiled BPT extension are built for that family, and `setup.py` uses no CPU-specific flags. So `pt_312_sam` should run as it is; Section 3 checks this.
- **Same limits:** every standard partition has a 6-hour limit, as on EPITO. The chained sbatch jobs therefore work the same way.

## 1. Partition and GPU names

From `sinfo` on mercurio (17 Nov 2026). Each partition also has a `-booked` twin for reservations.

| Partition | Nodes | GRES | Limit |
| --- | --- | --- | --- |
| `epito` | `epito[01-04]` | `gpu:a100:2` | 6 h |
| `gracehopper` | `gh[01-04]` | `gpu:gh200:1` | 6 h |
| `llm` | `llm-h200` | `gpu:h200:2` | 6 h |
| `cascadelake` | `cascadelake-L40S`, `cascadelake-V100S-*`, `cascadelake-T4` | `gpu:l40:2`, `gpu:v100:2`, `gpu:t4:2` | 6 h |
| `mirri` | `mirri-T4`, `mirri-L4` | `gpu:t4:3`, `gpu:l4:2` | 6 h |

Inside the shared `cascadelake` partition, the GPU type in the GRES selects the node (`--gres=gpu:l40:1` lands on the L40S node).

To see what is free right now:

```bash
sinfo -p gracehopper,llm -o "%20N %10T %35G"
squeue -p gracehopper,llm
```

## 2. Interactive session on the H100

```bash
srun -p gracehopper --gres=gpu:gh200:1 --cpus-per-task=16 --mem=64G -J "ShapBPT H100" --time=06:00:00 --pty bash
hostname
uname -m          # aarch64
nvidia-smi        # H100
export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
cd "$PROJECT_DIR"
```

## 3. Check the EPITO environment on the H100

Run each check in a fresh shell on the `gh` node:

```bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH            # As the sbatch files do
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, torch.__file__)
print("CUDA", torch.version.cuda, "available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
print("compiled for:", torch.cuda.get_arch_list())
print("CUDA calculation:", (torch.ones(4, device="cuda") + 1).cpu())
import shap_bpt
from shap_bpt import bpt
print("BPT extension:", bpt.__file__)
PY
```

**It works when:** the GPU shows as an H100 with capability `(9, 0)`, the calculation prints `tensor([2., 2., 2., 2.])`, and `shap_bpt` imports.

**If torch is missing**, or the H100 is rejected with "no kernel image is available": the environment relies on the node's pre-built PyTorch. Try the cluster's two locations in turn:

```bash
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages   # EPITO's path
# or
export PYTHONPATH=/opt/pytorch/lib/python3.12/site-packages          # path in the cluster documentation
ls /opt | grep -i torch                                              # what this node has
```

If one of them passes the check, note it for the sbatch jobs (Section 5). If none does, build a separate environment (Section 4).

**If only `shap_bpt` fails to import,** rebuild the extension once. It is the same aarch64 file EPITO uses, and the rebuilt file works on both:

```bash
python setup.py build_ext --inplace
```

Then check SAM itself:

```bash
python - <<'PY'
import os, torch
from sam2.build_sam import build_sam2
model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
                   os.path.join(os.environ["SAM2_DIR"], "checkpoints/sam2.1_hiera_large.pt"), device="cuda")
print("SAM-2.1 Large on", torch.cuda.get_device_name(0))
from segment_anything import sam_model_registry
sam_model_registry["vit_b"](checkpoint=os.path.join(os.environ["PROJECT_DIR"], "checkpoints/sam1/sam_vit_b_01ec64.pth")).to("cuda")
print("SAM-1 ViT-B OK")
PY
```

## 4. Only if Section 3 fails: a separate environment for the `gh` nodes

Use a new path so the EPITO environment stays untouched:

```bash
export SAM_ENV=/beegfs/home/mrashid/pt_312_sam_gh
python3.12 -m venv --system-site-packages "$SAM_ENV"     # sees the node's pre-built PyTorch
source "$SAM_ENV/bin/activate"
export PYTHONPATH=/opt/pytorch/lib/python3.12/site-packages   # whichever path Section 3 found
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then follow [EPITO_SETUP.md](EPITO_SETUP.md) Sections 3 to 5 with this environment active: ShapBPT with `pip install -e .`, SAM 1, and SAM 2 (`pip install -e "$SAM2_DIR"`). Skip Section 2 (cloning `pt_312`) and the checkpoint downloads, because the checkpoints are already on `/beegfs`. Repeat the checks in Section 3.

- **No internet on compute nodes?** Run the `pip install` lines on the login node with the same environment active. Packages are written to `/beegfs`, which the `gh` node also sees.
- **Do not reinstall `torch` with pip.** The pre-built `/opt` PyTorch matches the node's CUDA driver.

## 5. Run the usual jobs on the H100

The sbatch files ask for `--partition=epito --gres=gpu:a100:...`. Options given to `sbatch` on the command line override the `#SBATCH` lines in the file, so the same files run on a `gh` node:

```bash
mkdir -p logs
GH="-p gracehopper --gres=gpu:gh200:1 --cpus-per-task=16 --mem=64G"

# Partition generation, chained 6-hour jobs
sbatch $GH --export=ALL,CHAIN_LEFT=10 src/hpc/run_partitions.sbatch src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --device cuda --generate_at_least_n_masks 200 --save-partitions sam compact

# XAI at budgets 3000 and 5000, resumable
sbatch $GH --export=ALL,CHAIN_LEFT=40 src/hpc/run_xai_budgets.sbatch --budgets 3000 5000 -- \
  src/scripts/explanations/OD_yolo.py --config MSCOCO_epito_sam1 --show-gt --explain_all --no-heatmaps
```

- **Separate environment from Section 4:** add it to the export, `--export=ALL,SAM_ENV=/beegfs/home/mrashid/pt_312_sam_gh,CHAIN_LEFT=10`. Every sbatch file reads `SAM_ENV`, falling back to `pt_312_sam`.
- **`PYTHONPATH` needed in Section 3:** most sbatch files run `unset PYTHONPATH`. Use the separate environment from Section 4 instead, so the jobs do not depend on that variable.
- **`run_mscoco_large.sbatch` and `partitions_imagenet.sbatch`** pin `--nodelist=epito01`. Pass `--nodelist=gh01` (or `gh02`–`gh04`) as well, or the job waits for `epito01`.
- **Mixing nodes is safe.** EPITO and `gh` jobs can work on the same partition folder or XAI run. Both resume per image, and the generation settings do not include the machine. Give each job different images or datasets so they do not compute the same image twice.
- **One GPU per `gh` node.** `--gres=gpu:2` cannot be met there. For two GPUs at once, submit two jobs, which may land on two `gh` nodes.

Check jobs and logs as on EPITO:

```bash
squeue -u $USER
tail -f logs/partitions_<jobid>.out
```

## 6. Second choice: the L40S node (x86_64)

The L40S node has an Intel CPU, so nothing compiled for aarch64 runs there. It needs its own environment, the same `/beegfs/home/mrashid/pt_312_sam_x86` as the H200 node. Build it once as in Section 7, either on the H200 node or inside an L40S allocation:

```bash
srun -p cascadelake --gres=gpu:l40:1 --cpus-per-task=16 --mem=64G --time=06:00:00 --pty bash
uname -m                                        # x86_64
```

The repository can stay shared. The compiled extension is named per architecture (`bpt.cpython-312-x86_64-linux-gnu.so` next to `bpt.cpython-312-aarch64-linux-gnu.so`), so building it here does not break EPITO. Submit jobs with `-p cascadelake --gres=gpu:l40:1 --export=ALL,SAM_ENV=/beegfs/home/mrashid/pt_312_sam_x86`.

## 7. The H200 node (`llm` partition, x86_64)

The H200 node has the heaviest GPUs: 2 × H200 with 141 GB each. It is **x86_64**. With `pt_312_sam`, torch loads from the node's own `/opt/pytorch-v2.7.1`, but NumPy fails with `No module named 'numpy.core._multiarray_umath'`. That is the ARM NumPy of the EPITO environment. The node needs its own environment. Build it once. Start a session, **wait for the `mrashid@llm-h200` prompt**, then run the script:

```bash
srun -p llm --gres=gpu:h200:1 --cpus-per-task=16 --mem=64G --time=02:00:00 --pty bash
# ... wait for the prompt on llm-h200, then:
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
bash src/hpc/setup_x86_env.sh
```

Lines pasted while `srun` is still waiting do not reach the new shell, so paste nothing before the prompt appears. The script stops early on a non-x86 node or when `/opt/pytorch-v2.7.1` is missing, and it can be rerun. It runs the same steps as the manual commands below, followed by the check:

```bash
srun -p llm --gres=gpu:h200:1 --cpus-per-task=16 --mem=64G --time=02:00:00 --pty bash
uname -m                                   # x86_64
deactivate 2>/dev/null; unset PYTHONPATH
export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
export SAM_ENV=/beegfs/home/mrashid/pt_312_sam_x86
OPT_TORCH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages

# 1. Environment that sees the node's PyTorch, even when PYTHONPATH is unset (as in the sbatch files)
python3.12 -m venv "$SAM_ENV"
source "$SAM_ENV/bin/activate"
echo "$OPT_TORCH" > "$SAM_ENV/lib/python3.12/site-packages/opt_pytorch.pth"
python -m pip install --upgrade pip setuptools wheel Cython

# 2. Packages, built for x86_64. Never pip-install torch or torchvision: they come from /opt.
python -m pip install -r "$PROJECT_DIR/requirements-xai-all.txt"
python -m pip install PyYAML pycocotools scipy pandas pillow tabulate matplotlib transformers timm
python -m pip install --no-deps ultralytics
python -m pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
python -m pip install hydra-core iopath tqdm
SAM2_BUILD_CUDA=0 python -m pip install --no-deps -e "$SAM2_DIR"

# 3. ShapBPT: builds bpt.cpython-312-x86_64-linux-gnu.so next to EPITO's aarch64 file
cd "$PROJECT_DIR"
python -m pip install --no-deps -e .
```

`--no-deps` stops pip from replacing the node's torch with a generic wheel. Then check:

```bash
python - <<'PY'
import numpy, torch, torchvision, shap_bpt
from shap_bpt import bpt
print("numpy", numpy.__version__, "torch", torch.__version__, torch.__file__)
print("GPU:", torch.cuda.get_device_name(0), "x", torch.cuda.device_count())
print("CUDA calculation:", (torch.ones(4, device="cuda") + 1).cpu())
print("BPT extension:", bpt.__file__)
PY
```

It is ready when the GPU shows as an H200, the calculation prints `tensor([2., 2., 2., 2.])`, and the extension file ends in `x86_64-linux-gnu.so`. Then run the SAM check from Section 3. Keep this environment's `source` line, instead of `pt_312_sam`, for later sessions on this node.

- **`pip` cannot reach the internet:** run the `pip install` lines on the login node (`mercurio-login`) with the same environment active, provided `uname -m` there is also `x86_64`. Keep the last line (`pip install -e .`, which compiles the extension) and the check on the H200 node.
- **`pip check` complains about torch versions:** leave it. The `/opt` build is the one that matches the node's driver.

Submit jobs with this environment:

```bash
H200="-p llm --gres=gpu:h200:1 --cpus-per-task=16 --mem=64G"
sbatch $H200 --export=ALL,SAM_ENV=/beegfs/home/mrashid/pt_312_sam_x86,CHAIN_LEFT=10 \
  src/hpc/run_partitions.sbatch src/scripts/generate_coco_partitions.py --config MSCOCO_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" --device cuda \
  --generate_at_least_n_masks 200 --save-partitions sam compact
```

The same `pt_312_sam_x86` environment also serves the L40S node (Section 6), which is x86_64 too. That works if `/opt/pytorch-v2.7.1` exists there as well; check with `ls /opt` on that node. The partition name suggests the node is meant for LLM jobs. If others are waiting, prefer `gracehopper`.

## Results are comparable across GPUs

The partitions and attributions are the same up to floating-point differences. Timings are not. When reporting generation or explanation times, note the GPU. `*_results.json` records only `cuda` as the device; the job log's first line names the node (`Job ... on <host>`), and a `gh` host means H100. Do not mix A100 and H100 times in one timing table without saying so.
