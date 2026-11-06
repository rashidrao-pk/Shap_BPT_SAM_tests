# BUSI SigLIP

| | |
| --- | --- |
| Used with | [BUSI](../DATASETS/BUSI.md) |
| Task | Classification: benign, malignant, normal |
| Runner | `OC_BUSI` |
| Input | Direct resize to 224×224, mean/std 0.5 |
| Checkpoint | `checkpoints/busi_siglip` |

## Download

```bash
python - <<'PY'
from pathlib import Path
from transformers import AutoModelForImageClassification
name = 'ghostsas001/busi-breast-ultrasound-siglip'
folder = Path('checkpoints/busi_siglip')
AutoModelForImageClassification.from_pretrained(name).save_pretrained(folder)
print('Saved:', folder.resolve())
PY
```

## Why this model

It is a public classifier for the same three classes as the dataset, with documented preprocessing.

## Limits

- It was trained on BUSI, so the explained images are not held out.
- Its model card warns that duplicates may cross the train/test split.
- LRP is not available for this architecture.
- Research diagnostics only, not clinical evidence.

More detail: [SigLIP notes](../Partitions/MODELS/SigLIP_BUSI.md).
