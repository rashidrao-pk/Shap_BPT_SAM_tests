# BUSI: one-time downloads on EPITO

Run from the repository root with your existing environment activated:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
```

For environment and SAM package installation, use [EPITO setup](../Partitions/full/EPITO_SETUP.md). Download files on a node with internet access; inference can use local copies afterward.

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

## Models

- [BUSI SigLIP](../MODELS/BUSI_SIGLIP.md)

## Next steps

1. [Generate partitions](../Partitions/DATASETS/BUSI.md)
2. [Run explanations](../XAI/DATASETS/BUSI.md)
