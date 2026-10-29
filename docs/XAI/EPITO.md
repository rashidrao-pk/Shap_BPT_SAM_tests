# EPITO: XAI from generated partitions

First generate partitions using [Partitions/EPITO.md](../Partitions/EPITO.md).
Generation and XAI are separate commands; XAI reuses the saved label maps.

## 1. Start a session

If your partition-generation GPU session is still active, reuse it.
Otherwise:

```bash
ssh epito-mercurio
srun -p epito --gres=gpu:a100:1 -J "ShapBPT" --time=06:00:00 --pty bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages
export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
cd "$PROJECT_DIR"
```

## 2. Check the config

[src/configs/MSCOCO_epito.yaml](../../src/configs/MSCOCO_epito.yaml) selects:

```yaml
data:
  masks_path: /beegfs/home/mrashid/datasets/
  mask_dir: SAM_BPT
  mask_dir_final: MSCOCO_sam2_tiny_62_refined_200
evaluation:
  methods: [BPT, SAM_c, SAM_c_kernelshap]
```

These are the relevant fields, not a replacement for the complete config.
`OD_yolo.py` defaults to BPT, BPT+SAM_c, and SAM_c+Kernel SHAP.
Both SAM_c methods require the saved `*_compact.npy` maps.
Optional notebook stages such as `boosted` are available only when explicitly saved
during generation. See [the YOLO explanation guide](OD_yolo.md).

## 3. Run two images

```bash
python src/scripts/explations/OD_yolo.py \
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

The runner discovers generated image IDs from the configured partition folder
and selects the first two in sorted order. Confirm the printed `Masks`,
`methods`, and `Images to process` values. Result paths are printed at startup.
The new script ignores legacy method lists and defaults to the three SAM_c comparison methods.

## 4. Run more images

Increase `--limit`, or omit it to process all image IDs found in the partition
folder. Generate those partitions first. For other models and experiment
settings, see [the experiment matrix](../README_exp_epito.md).

Use a distinct output directory when comparing partition datasets, for example
add `--output-dir outputs/MSCOCO_sam2_tiny_62_refined_200`. This avoids mixing results with older
runs. Use `exit` when finished to release the GPU allocation.
