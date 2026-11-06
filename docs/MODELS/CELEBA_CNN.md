# CelebA CNN

| | |
| --- | --- |
| Used with | [CelebA-HQ](../DATASETS/CelebAHQ.md) |
| Task | 40 independent facial attributes |
| Runner | `OC_CelebA` |
| Checkpoint | `src/checkpoints/celeba_model.pth` (42 MB) |

## Download

```bash
python src/scripts/download_celeba_checkpoint.py
```

The script checks the SHA-256. `CelebAHQ_epito.yaml` already points to this file.

## Why this model

It is the model from the [reference E5 notebook](https://github.com/rashidrao-pk/shap_bpt_tests/blob/main/notebooks/E5_CelebA/N1_Run_experiments_CelebA.ipynb), and many attributes map directly to annotated face parts.

## Limits

XAI explains the raw attribute logit, not a softmax probability, so scores are not comparable with the classifiers here.

Usage: [CelebA XAI](../XAI/DATASETS/CELEBA.md).
