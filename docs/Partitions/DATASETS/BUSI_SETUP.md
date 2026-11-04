# BUSI: one-time downloads on EPITO

Run from the repository root with your existing environment activated:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
```

For environment and SAM package installation, use [EPITO setup](../full/EPITO_SETUP.md). Download files on a node with internet access; inference can use local copies afterward.

## 1. Images and lesion masks

Use the [Kaggle BUSI mirror](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). The older Cairo University URL can return a web page or an incomplete file; a `.zip` filename does not guarantee a valid archive.

Try the public download endpoint, saving to a new filename so any previous download is retained:

```bash
mkdir -p /beegfs/home/mrashid/datasets/BUSI
curl --fail --location --retry 3 \
  https://www.kaggle.com/api/v1/datasets/download/aryashah2k/breast-ultrasound-images-dataset \
  --output /beegfs/home/mrashid/datasets/BUSI/BUSI_kaggle.zip
unzip -tq /beegfs/home/mrashid/datasets/BUSI/BUSI_kaggle.zip && \
  unzip -n /beegfs/home/mrashid/datasets/BUSI/BUSI_kaggle.zip \
    -d /beegfs/home/mrashid/datasets/BUSI
```

If Kaggle returns a login/security page, download using its website while signed in and transfer the ZIP to EPITO. The public endpoint is not guaranteed to work without authentication. Always run `unzip -tq` before extracting. Keep annotation PNG files alongside images.

To inspect an unsuccessful download:

```bash
file /beegfs/home/mrashid/datasets/BUSI/Dataset_BUSI.zip
ls -lh /beegfs/home/mrashid/datasets/BUSI/Dataset_BUSI.zip
```

HTML/text output indicates a web response rather than dataset data. A file identified as ZIP that fails `unzip -tq` is damaged or incomplete and needs downloading again.

Set both paths in `src/configs/BUSI_epito.yaml` to the extracted folder directly containing `benign/`, `malignant/`, and `normal/`. If the extracted directory is `Dataset_BUSI_with_GT`, update these fields while retaining the other config settings:

```yaml
data:
  image_dir: /beegfs/home/mrashid/datasets/BUSI/Dataset_BUSI_with_GT
  ground_truth:
    format: busi
    root: /beegfs/home/mrashid/datasets/BUSI/Dataset_BUSI_with_GT
```

Check image counts, excluding annotations, from the repository root:

```bash
python - <<'PY'
from collections import Counter
from pathlib import Path
from src.medical.datasets import discover_images
root = Path('/beegfs/home/mrashid/datasets/BUSI/Dataset_BUSI_with_GT')
records = discover_images(root)
print('Images:', len(records))
print(dict(Counter(Path(item['file_name']).parent.name for item in records)))
PY
```

If extraction has a different nesting level, adjust the config paths and the check above.

## 2. SigLIP classifier

Follow [SigLIP download instructions](../MODELS/SigLIP_BUSI.md), including how to save a local checkpoint folder and load it offline. This classifier is separate from the SAM segmentation checkpoint.

## 3. SAM checkpoint

Skip checkpoints you already have. SAM packages must already be installed through the EPITO setup guide.

SAM2.1 Tiny, matching the routine BUSI command ([Meta download script](https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh)):

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
mkdir -p "$SAM2_DIR/checkpoints"
curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  --output "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt"
```

Alternatively, SAM1 ViT-B ([Meta checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)):

```bash
mkdir -p checkpoints/sam1
curl --fail --location --retry 3 \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  --output checkpoints/sam1/sam_vit_b_01ec64.pth
```

Next: [generate BUSI partitions](BUSI.md), then [run XAI](../../XAI/BUSI.md). Downloads do not need repeating for each experiment.
