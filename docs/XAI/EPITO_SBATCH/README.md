# XAI batch jobs on EPITO

Submit from the repository root on `mercurio-login`. Each command submits one experiment with a six-hour GPU allocation. The job continues after you disconnect.

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
mkdir -p logs
```

| Dataset                 | Task                                            | Model                    | Runner                |
| ----------------------- | ----------------------------------------------- | ------------------------ | --------------------- |
| ImageNet + ImageNetS GT | Object classification (OC)                      | ResNet50                 | `OC_ResNet.py`        |
| ImageNet + ImageNetS GT | OC                                              | VGG16                    | `OC_VGG.py`           |
| ImageNet + ImageNetS GT | OC                                              | ViT-B/16                 | `OC_ViT.py`           |
| MS-COCO                 | Object detection (OD)                           | YOLO11s                  | `OD_yolo.py`          |
| MS-COCO                 | OD                                              | DETR / ResNet50 backbone | `OD_DETR.py`          |
| CAMO                    | Classification explanation on camouflage images | ViT-B/16                 | `COD_ViT.py`          |
| CelebA-HQ               | Attribute classification                        | CelebModel CNN           | `OC_CelebA.py`        |
| BUSI                    | Benign/malignant/normal classification          | SigLIP                   | `run_busi_shapbpt.py` |
| PartImageNet            | OC with part GT                                 | ResNet50                 | `OC_PartImageNet.py`  |

The available transformer **detector** is [DETR](https://huggingface.co/facebook/detr-resnet-50), which combines a ResNet50 backbone with a transformer. `OC_ViT.py` is a classifier, not an object detector. A ViT-backbone detector is not currently implemented.

## Before submitting

Generate partitions and set `data.mask_dir_final` in each dataset config to the desired SAM folder. The shared XAI runners require saved `compact` partitions (displayed as SAM_c). Prepare the model checkpoints before submitting if compute nodes cannot access the internet.

For ImageNet classifiers, cache the pretrained weights once using the working EPITO environment on a node with internet access:

```bash
python - <<'PY'
from torchvision.models import ResNet50_Weights, VGG16_Weights, ViT_B_16_Weights
for weights in [ResNet50_Weights.DEFAULT, VGG16_Weights.DEFAULT, ViT_B_16_Weights.DEFAULT]:
    weights.get_state_dict(progress=True, check_hash=True)
PY
```

These are classification weights; VGG16 uses [IMAGENET1K_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html). A local compatible checkpoint can be supplied with `--checkpoint` instead. Keep the matching `--weights` preprocessing choice.

For other tasks, use the checkpoint setup in [CelebA](../CELEBA.md), [BUSI](../../Partitions/DATASETS/BUSI_SETUP.md), and [PartImageNet](../../Partitions/DATASETS/PARTIMAGENET_SETUP.md). Cache `facebook/detr-resnet-50` and YOLO weights before submitting if needed. All-method runs require the optional dependencies described in [ALL_METHODS.md](../ALL_METHODS.md).

## ImageNet: ResNet50

```bash
sbatch --job-name=IN-ResNet src/hpc/run_xai.sbatch \
  src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only \
  --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

## ImageNet: VGG16

```bash
sbatch --job-name=IN-VGG16 src/hpc/run_xai.sbatch \
  src/scripts/explanations/OC_VGG.py \
  --config ImageNet_epito --show-gt --gt-only \
  --weights IMAGENET1K_V1 --model-batch-size 4 \
  --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

## ImageNet: ViT-B/16

```bash
sbatch --job-name=IN-ViT src/hpc/run_xai.sbatch \
  src/scripts/explanations/OC_ViT.py \
  --config ImageNet_epito --show-gt --gt-only \
  --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

## MS-COCO: YOLO11s

```bash
sbatch --job-name=COCO-YOLO src/hpc/run_xai.sbatch \
  src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito --model yolo11s --show-gt \
  --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

## MS-COCO: DETR

```bash
sbatch --job-name=COCO-DETR src/hpc/run_xai.sbatch \
  src/scripts/explanations/OD_DETR.py \
  --config MSCOCO_epito --model facebook/detr-resnet-50 --show-gt \
  --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

Add `--local-files-only` after the DETR model is cached if you want loading to require local files.

## CAMO: ViT-B/16

```bash
sbatch --job-name=CAMO-ViT src/hpc/run_xai.sbatch \
  src/scripts/explanations/COD_ViT.py \
  --config CAMO_epito --show-gt \
  --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

By default this is an ImageNet classifier evaluated on CAMO images, not a trained camouflage segmentation model. For a compatible CAMO-trained ViT classifier, add `--checkpoint /path/to/model.pt` and provide its class names with `--class-names /path/to/classes.json` if they are not stored in the checkpoint.

## CelebA-HQ: attribute CNN

```bash
sbatch --job-name=CelebA-Hair src/hpc/run_xai.sbatch \
  src/scripts/explanations/OC_CelebA.py \
  --config CelebAHQ_epito --attribute Brown_Hair --show-gt \
  --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

Change `Brown_Hair` to the attribute to explain, for example `Eyeglasses`. The model checkpoint comes from `model.checkpoint` in the config. Attribute experiments have separate result folders.

## BUSI: SigLIP with SAM1 and SAM2

Use the separate BUSI pipeline with the classifier already downloaded to `checkpoints/busi_siglip`:

```bash
sbatch --job-name=BUSI-SigLIP src/hpc/run_xai.sbatch \
  src/scripts/run_busi_shapbpt.py run \
  --model-dir checkpoints/busi_siglip --limit 780 \
  --sam sam1 sam2 --regions 62 --budget 1000 --batch-size 8 \
  --sam1-checkpoint checkpoints/sam1/sam_vit_b_01ec64.pth \
  --sam2-checkpoint /beegfs/home/mrashid/repos/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt
```

This pipeline generates/reuses its own SAM partitions and BPT trees, explains the predicted class, and saves under `results/busi`. It runs its hierarchical SAM/BPT method for both SAM variants; it does not accept `--explain_all`, `--no-heatmaps`, or the shared runner's AUC options. Its budget is rounded to complete hierarchical permutations. For the separate three-method shared workflow, see [BUSI explanations](../BUSI.md).

## PartImageNet: ResNet50

```bash
sbatch --job-name=PartIN-ResNet src/hpc/run_xai.sbatch \
  src/scripts/explanations/OC_PartImageNet.py \
  --config PartImageNet_epito --show-gt \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 \
  --explain_all --budget 1000 --no-heatmaps --auc-num-samples 200
```

Part GT follows the selected annotation JSON; `--gt-only` is unnecessary for this runner.

## Run each model with SAM1 and SAM2 partitions

Each shared-runner command uses **one** `data.mask_dir_final`. To run both variants without changing a queued job's config, make two config copies before submission. For example:

```bash
cp src/configs/ImageNet_epito.yaml src/configs/ImageNet_sam1_epito.yaml
cp src/configs/ImageNet_epito.yaml src/configs/ImageNet_sam2_epito.yaml
```

Set `data.mask_dir_final` in the first copy to your completed SAM1 folder and in the second to your SAM2 folder. Keep `data.name` and GT settings unchanged. Submit each model command twice, replacing its config with `ImageNet_sam1_epito` and `ImageNet_sam2_epito`. Use the same approach for MS-COCO, CelebA-HQ, CAMO, and PartImageNet.

Output folders include the runner and partition folder, so different classifiers and SAM variants remain separate. Do not submit two jobs with identical settings writing the same output directory.

## Small test, monitoring, and resume

For a small test, add `--limit 1` to a shared-runner command. For BUSI, change `--limit 780` to `--limit 2`. Budgets shown above are examples; use your validated budget selection for production experiments.

```bash
squeue -u mrashid
tail -f logs/XAI_JOBID.out logs/XAI_JOBID.err
python src/scripts/check_experiment_progress.py --results-root results/explanations
```

Replace `JOBID` with the submitted job ID. The separate BUSI pipeline reports its own progress under `results/busi`.

After timeout, submit the identical command again. Completed images are skipped; an interrupted image repeats its XAI. Keep `--no-resume` out of continuation commands. `--no-heatmaps` suppresses individual map PNGs while retaining combined plots. AUC is enabled in the shared examples with 200 requested points. Architecture-specific methods that are unsupported are reported explicitly. See [resume details](../RESUME_EPITO.md).
