# BUSI SigLIP checkpoint

The [BUSI SigLIP model](https://huggingface.co/ghostsas001/busi-breast-ultrasound-siglip) is a three-class image classifier: index 0 benign, 1 malignant, 2 normal. The runner reads labels from the checkpoint and uses the model-card preprocessing, freezes its weights, and explains the original image's predicted class using softmax probability. It does not train the model.

Use the existing Transformers installation in your SAM environment. Download once to the Hugging Face cache (no GPU needed):

```bash
python - <<'PY'
from transformers import AutoModelForImageClassification
name = 'ghostsas001/busi-breast-ultrasound-siglip'
model = AutoModelForImageClassification.from_pretrained(name)
print(model.config.id2label)
PY
```

For a named checkpoint folder instead of relying on the cache, download once from the repository root:

```bash
python - <<'PY'
from pathlib import Path
from transformers import AutoModelForImageClassification
name = 'ghostsas001/busi-breast-ultrasound-siglip'
folder = Path('checkpoints/busi_siglip')
model = AutoModelForImageClassification.from_pretrained(name)
model.save_pretrained(folder)
print('Saved:', folder.resolve())
print('Classes:', model.config.id2label)
PY
```

Use this local copy for subsequent experiments:

```bash
python src/scripts/explanations/OC_BUSI.py \
  --config BUSI_epito --checkpoint checkpoints/busi_siglip \
  --local-files-only --show-gt --limit 2 --budget 1000 --no-heatmaps
```

Routine runs can use `--local-files-only`. Alternatively supply `--checkpoint /path/to/local/huggingface/model_directory` containing the weights and model config. The shared runner applies the model-card transforms, so an image processor config is unnecessary.

The [MMFM checkpoints](https://huggingface.co/noe95/mmfm-breast-imaging-checkpoints) use a separate binary ResNet18 classifier and require their source architecture/preprocessing. They are not interchangeable with this SigLIP runner.

The SigLIP model card describes an image-level BUSI split with possible duplicate leakage. Treat BUSI results as research diagnostics; they do not establish independent patient-level performance.
