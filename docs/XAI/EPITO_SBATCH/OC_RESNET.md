```bash
mkdir -p logs

sbatch src/hpc/run_xai.sbatch \
  src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only \
  --explain_all --budget 1000 --no-heatmaps
```
