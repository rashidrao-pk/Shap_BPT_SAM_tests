# BUSI SigLIP checkpoint

The [BUSI SigLIP model](https://huggingface.co/ghostsas001/busi-breast-ultrasound-siglip) is a three-class image classifier: index 0 benign, 1 malignant, 2 normal. The runner reads labels and preprocessing from the checkpoint, freezes its weights, and explains the original image's predicted class using softmax probability. It does not train the model.

Install Transformers in your existing environment without replacing the site's PyTorch. Download once to the Hugging Face cache:

```bash
python - <<'PY'
from transformers import AutoImageProcessor, AutoModelForImageClassification
name = 'ghostsas001/busi-breast-ultrasound-siglip'
processor = AutoImageProcessor.from_pretrained(name)
model = AutoModelForImageClassification.from_pretrained(name)
print(model.config.id2label)
PY
```

Routine runs can use `--local-files-only`. Alternatively supply `--checkpoint /path/to/local/huggingface/model_directory` containing the weights, model config, and image processor config.

The [MMFM checkpoints](https://huggingface.co/noe95/mmfm-breast-imaging-checkpoints) use a separate binary ResNet18 classifier and require their source architecture/preprocessing. They are not interchangeable with this SigLIP runner.

The SigLIP model card describes an image-level BUSI split with possible duplicate leakage. Treat BUSI results as research diagnostics; they do not establish independent patient-level performance.
