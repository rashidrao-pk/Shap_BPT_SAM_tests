<!-- ImageNet -->
<!-- --------------------------------------------------------------- -->

#### ImageNet

```bash
mkdir -p logs

sbatch src/hpc/run_xai.sbatch \
  src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only \
  --explain_all --budget 1000 --no-heatmaps
```

---

<!-- --------------------------------------------------------------- -->

```bash
mkdir -p logs


```

<!-- MS-COCO -->
<!-- --------------------------------------------------------------- -->

#### MS-COCO

```bash
mkdir -p logs

python src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito \
  --limit 2 \
  --max-evals 50 \
  --kernel-shap-samples 50 \
```

<!-- Celeb-A-HQ -->
<!-- --------------------------------------------------------------- -->

#### Celeb-A-HQ

```bash
mkdir -p logs

python src/scripts/explanations/OC_CelebA.py \
  --config CelebAHQ_epito --attribute Eyeglasses \
  --show-gt --limit 1 --explain_all --no-heatmaps
```

<!-- BUSI -->
<!-- --------------------------------------------------------------- -->

#### BUSI

```bash
mkdir -p logs

python src/scripts/explanations/OC_BUSI.py \
  --config BUSI_epito --show-gt --limit 2 \
  --budget 1000 --no-heatmaps --local-files-only
```

<!-- --------------------------------------------------------------- -->
<!-- PARTNET -->

#### PARTNET

```bash
mkdir -p logs

python src/scripts/explanations/OC_PartImageNet.py \
  --config PartImageNet_epito \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth \
  --weights IMAGENET1K_V2 --show-gt --limit 2 \
  --budget 1000 --no-heatmaps
```
