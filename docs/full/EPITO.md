# EPITO: start a new SAM session

Installation is complete. Use these steps each time you reconnect. The
[one-time setup guide](EPITO_SETUP.md) is only needed when rebuilding the environment.

## 1. Connect from your local terminal

```bash
ssh epito-mercurio
```

## 2. Request a GPU on EPITO

```bash
srun -p epito --gres=gpu:a100:1 -J "ShapBPT SAM" --time=06:00:00 --pty bash
```

Wait for the compute-node shell before continuing.

## 3. Activate the existing SAM environment

```bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages
export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
export SAM2_DIR=/beegfs/home/mrashid/repos/XAI/third_party/sam2
cd "$PROJECT_DIR"
```

Keep this `PYTHONPATH`: it selects EPITO's working torch/torchvision stack.
The existing `src/scripts/prepare_setup.sh` activates the original `pt_312`,
so use the commands above for `pt_312_sam`. A cloned prompt may still display
`(pt_312)`; check the interpreter path below.

## 4. Check the session

```bash
python - <<'PY'
import sys
import torch
import torchvision
from segment_anything import sam_model_registry
from sam2.build_sam import build_sam2

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
assert torch.cuda.is_available(), "Check the Slurm GPU allocation"
print("GPU:", torch.cuda.get_device_name(0))
print("SAM1 and SAM2 imports OK")
PY
```

The Python path should be `/beegfs/home/mrashid/pt_312_sam/bin/python`.

## 5. Run your work

Run your SAM scripts from `$PROJECT_DIR`. Installed packages and downloaded
checkpoints persist; environment creation, installation, and downloads do not
need to be repeated. The existing checkpoint paths are:

- SAM1 ViT-B: `$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth`
- SAM2.1 Tiny: `$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt`
- SAM2.1 Tiny config: `configs/sam2.1/sam2.1_hiera_t.yaml`

For a small XAI experiment, after preparing the dataset and partition files
configured in [MSCOCO_epito.yaml](../../src/configs/MSCOCO_epito.yaml):

```bash
python src/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --max-evals 50 \
  --eval-batch-size 16 \
  --auc-batch-size 8 \
  --verbose-level medium \
  --model yolo11s \
  --bg-type noise \
  --class-aggregation sum
```

See the [EPITO experiment guide](../README_exp_epito.md) for other models.

## 6. Finish

After your scripts finish, release the interactive GPU allocation:

```bash
exit
```

This returns to the login shell. Next time, repeat steps 1–3 and run your work.
