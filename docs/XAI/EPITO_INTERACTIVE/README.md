# Interactive XAI runs on EPITO

Run each experiment directly with `python -u` to see terminal progress and errors as they happen. This guide mirrors the [batch-job guide](../EPITO_SBATCH/README.md), using the same datasets and models.

## Start an interactive GPU session

From `mercurio-login`, request a GPU shell (skip `srun` if you already have an allocated EPITO GPU shell):

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
srun -p epito --gres=gpu:a100:1 --cpus-per-task=12 --mem=64G \
  --time=06:00:00 --job-name=XAI-interactive --pty bash
```

Inside that GPU shell:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
export HF_HUB_DISABLE_XET=1
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
export PYTHONUNBUFFERED=1
nvidia-smi
```

The commands below start with one image (`--limit 1`); BUSI also starts with one. Remove the limit from shared runners for all completed partition images, or set a larger value. Figures are saved to files; interactive here means live terminal output, not GUI plot windows.

| Dataset                 | Task                                            | Model                    | Runner               |
| ----------------------- | ----------------------------------------------- | ------------------------ | -------------------- |
| ImageNet + ImageNetS GT | Object classification (OC)                      | ResNet50                 | `OC_ResNet.py`       |
| ImageNet + ImageNetS GT | OC                                              | VGG16                    | `OC_VGG.py`          |
| ImageNet + ImageNetS GT | OC                                              | ViT-B/16                 | `OC_ViT.py`          |
| MS-COCO                 | Object detection (OD)                           | YOLO11s                  | `OD_yolo.py`         |
| MS-COCO                 | OD                                              | DETR / ResNet50 backbone | `OD_DETR.py`         |
| CAMO                    | Classification explanation on camouflage images | ViT-B/16                 | `COD_ViT.py`         |
| CelebA-HQ               | Attribute classification                        | CelebModel CNN           | `OC_CelebA.py`       |
| BUSI                    | Benign/malignant/normal classification          | SigLIP                   | `OC_BUSI.py`         |
| PartImageNet            | OC with part GT                                 | ResNet50                 | `OC_PartImageNet.py` |

The available transformer **detector** is [DETR](https://huggingface.co/facebook/detr-resnet-50), which combines a ResNet50 backbone with a transformer. `OC_ViT.py` is a classifier, not an object detector. A ViT-backbone detector is not currently implemented.

## Before running

Generate partitions and set `data.mask_dir_final` in each dataset config to the desired SAM folder. The shared XAI runners require saved `compact` partitions (displayed as SAM_c). Prepare the model checkpoints before running if compute nodes cannot access the internet.

For ImageNet classifiers, cache the pretrained weights once using the working EPITO environment on a node with internet access:

```bash
python - <<'PY'
from torchvision.models import ResNet50_Weights, VGG16_Weights, ViT_B_16_Weights
for weights in [ResNet50_Weights.DEFAULT, VGG16_Weights.DEFAULT, ViT_B_16_Weights.DEFAULT]:
    weights.get_state_dict(progress=True, check_hash=True)
PY
```

These are classification weights; VGG16 uses [IMAGENET1K_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html). A local compatible checkpoint can be supplied with `--checkpoint` instead. Keep the matching `--weights` preprocessing choice.

For other tasks, use the checkpoint setup in [CelebA](../CELEBA.md), [BUSI](../../Partitions/DATASETS/BUSI_SETUP.md), and [PartImageNet](../../Partitions/DATASETS/PARTIMAGENET_SETUP.md). Cache `facebook/detr-resnet-50` and YOLO weights before running if needed. All-method runs require the optional dependencies described in [ALL_METHODS.md](../ALL_METHODS.md).

## ImageNet: ResNet50

```bash
python -u src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only \
  --limit 1 --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

## ImageNet: VGG16

```bash
python -u src/scripts/explanations/OC_VGG.py \
  --config ImageNet_epito --show-gt --gt-only \
  --weights IMAGENET1K_V1 --model-batch-size 4 \
  --limit 1 --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

## ImageNet: ViT-B/16

```bash
python -u src/scripts/explanations/OC_ViT.py \
  --config ImageNet_epito --show-gt --gt-only \
  --limit 1 --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

## MS-COCO: YOLO11s

```bash
python -u src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito --model yolo11s --show-gt \
  --limit 1 --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

## MS-COCO: DETR

```bash
python -u src/scripts/explanations/OD_DETR.py \
  --config MSCOCO_epito --model facebook/detr-resnet-50 --show-gt \
  --limit 1 --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

Add `--local-files-only` after the DETR model is cached if you want loading to require local files.

## CAMO: ViT-B/16

```bash
python -u src/scripts/explanations/COD_ViT.py \
  --config CAMO_epito --show-gt \
  --limit 1 --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

By default this is an ImageNet classifier evaluated on CAMO images, not a trained camouflage segmentation model. For a compatible CAMO-trained ViT classifier, add `--checkpoint /path/to/model.pt` and provide its class names with `--class-names /path/to/classes.json` if they are not stored in the checkpoint.

## CelebA-HQ: attribute CNN

```bash
python -u src/scripts/explanations/OC_CelebA.py \
  --config CelebAHQ_epito --attribute Brown_Hair --show-gt \
  --limit 1 --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

Change `Brown_Hair` to the attribute to explain, for example `Eyeglasses`. The model checkpoint comes from `model.checkpoint` in the config. Attribute experiments have separate result folders.

## BUSI: SigLIP with all methods

Set `BUSI_epito.yaml` to the saved compact partition folder, then run the same shared comparison as the other datasets:

```bash
python -u src/scripts/explanations/OC_BUSI.py \
  --config BUSI_epito --checkpoint checkpoints/busi_siglip --show-gt \
  --limit 1 --explain_all --budget 1000 --no-heatmaps \
  --auc-num-samples 200 --local-files-only --bg-type full
```

Repeat with a SAM1 or SAM2 config copy to compare both partition sources. The existing `run_busi_shapbpt.py run --explain_all --sam sam1` command can also select saved partitions and dispatch to this shared runner. See [BUSI all-method instructions](../BUSI.md). SigLIP GradCAM is supported; transformer LRP is reported unsupported.

## PartImageNet: ResNet50

```bash
python -u src/scripts/explanations/OC_PartImageNet.py \
  --config PartImageNet_epito --show-gt \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 \
  --limit 1 --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

Part GT follows the selected annotation JSON; `--gt-only` is unnecessary for this runner.

## Run each model with SAM1 and SAM2 partitions

Each shared-runner command uses **one** `data.mask_dir_final`. To run both variants while keeping each run's config separate, make two config copies before starting. For example:

```bash
cp src/configs/ImageNet_epito.yaml src/configs/ImageNet_sam1_epito.yaml
cp src/configs/ImageNet_epito.yaml src/configs/ImageNet_sam2_epito.yaml
```

Set `data.mask_dir_final` in the first copy to your completed SAM1 folder and in the second to your SAM2 folder. Keep `data.name` and GT settings unchanged. Run each model command twice, replacing its config with `ImageNet_sam1_epito` and `ImageNet_sam2_epito`. Use the same approach for MS-COCO, CelebA-HQ, CAMO, and PartImageNet.

Output folders include the runner and partition folder, so different classifiers and SAM variants remain separate. Do not start two processes with identical settings writing the same output directory.

## Watch progress and resume

`python -u` streams output directly to the terminal. Run another command after the current experiment finishes, or use a separate allocated GPU session for another experiment. Budgets shown above are examples; use your validated budget selection for production experiments.

In another terminal, inspect shared-runner progress:

```bash
python src/scripts/check_experiment_progress.py --results-root results/explanations
```

Shared BUSI runs write under the configured explanation/evaluation roots and are included in the progress checker. Use the result paths printed at the end of each command to inspect the saved figures and metrics.

To stop a foreground experiment, press Ctrl+C. Rerun the **identical command** to continue: completed images are skipped, while an interrupted image repeats its XAI. Keep `--no-resume` out of continuation commands. A limit includes already completed images; increase or remove it to select more images.

For example, continue with up to ten ImageNet images:

```bash
python -u src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only \
  --limit 10 --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

Terminal disconnection can stop an interactive run, and the GPU allocation still has its time limit. For unattended runs use the [batch-job guide](../EPITO_SBATCH/README.md). See [resume details](../RESUME_EPITO.md) for checkpoint boundaries and settings compatibility.
