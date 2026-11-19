```bash
watch -n 30 'squeue -p llm,gracehopper,epito,cascadelake,mirri -t RUNNING \
  -o "%.12P %.20N %.10u %.15b %.10M %.10L %.20j" -S P,N'

cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
watch -n 30 bash src/hpc/gpu_check.sh --no-small
```
