## SAM 2.1

Model Options:

- [x] sam2.1_hiera_tiny
- [ ] sam2.1_hiera_small
- [ ] sam2.1_hiera_large

### Tiny Model

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --limit 500
```

### Large Model

```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --limit 500
```

#### On EPITO, submit from the login node:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
mkdir -p logs
sbatch src/hpc/run_mscoco_large.sbatch
```

#### Check progress:

```bash
squeue -u mrashid

tail -f /beegfs/home/mrashid/datasets/SAM_BPT/MSCOCO_sam2_large_62_refined_200.log
```
