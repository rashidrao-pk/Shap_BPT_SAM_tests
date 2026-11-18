# BUSI partitions on EPITO

- [Download the dataset](../../DATASETS/BUSI.md)
- Download checkpoints: [SAM-1](../../MODELS/SAM1.md), [SAM-2](../../MODELS/SAM2.md)

First time: [download the dataset, classifier, and SAM checkpoints](BUSI_SETUP.md). The commands below are for routine generation after setup.

## SAM1

`--compact-strategy greedy split` saves both compact selections from one SAM run: `<id>_compact_greedy.npy` and `<id>_compact_split.npy` in the same folder. `data.compact_strategy` in the config picks the one used by XAI (default `greedy`); see [MSCOCO.md](MSCOCO.md#compare-compact-strategies-greedy-vs-mso).

```bash
python src/scripts/generate_busi_partitions.py \
  --config BUSI_epito \
  --model sam1 \
  --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b \
  --device cuda --max-regions 254 --generate_at_least_n_masks 50 \
  --save-partitions sam compact --compact-strategy greedy split --limit 1000
```

## SAM2

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_busi_partitions.py \
  --config BUSI_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda --max-regions 254 --generate_at_least_n_masks 50 \
  --save-partitions sam compact --compact-strategy greedy split --limit 1000
```

#### Large Model

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_busi_partitions.py \
  --config BUSI_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda --max-regions 254 --generate_at_least_n_masks 50 \
  --save-partitions sam compact --compact-strategy greedy split --limit 1000
```

Next: [SigLIP checkpoint](../MODELS/SigLIP_BUSI.md) and [BUSI explanations](../../XAI/DATASETS/BUSI.md).

## Preview saved partitions

```bash
python src/scripts/view_partitions.py \
  --config BUSI_epito \
  --limit 2 \
  --methods sam compact
```

## Check progress

```bash
python src/scripts/check_partition_progress.py --limit 1000
```

The default summary includes BUSI and PartImageNet. BUSI excludes annotation PNGs; PartImageNet follows the configured annotation split. Missing dataset/config paths are reported explicitly. BUSI totals stop at the available image count (780 for the full release), even with `--limit 1000`.

Partitions from `run_busi_shapbpt.py` under `results/busi/<case>/sam1|sam2` are also included as separate rows when present. To inspect a custom location:

```bash
python src/scripts/check_partition_progress.py --config BUSI_epito \
  --busi-results-root results/busi --limit 1000
```

Nested BUSI sizes include SAM `input_*` artifacts, excluding attribution maps and XAI plots. Recorded time is SAM inference plus refinement, excluding XAI time. `--configured-only` restricts the summary to the flat folder in `data.mask_dir_final`.

To keep **all batch SAM1/SAM2 variants** while hiding the separate BUSI pipeline/test rows, use:

```bash
python src/scripts/check_partition_progress.py \
  --limit 1000 --gt-only --no-busi-pipeline
```

Do not add `--configured-only` to this command: that option hides every variant except the single folder specified in each config. Hiding rows does not delete computed files.

### Medical priors: MedSAM and MedicoSAM

Same settings, with a medical SAM instead. Setup, downloads and how they work: [MEDICAL_SEGMENTERS.md](../SAM/MEDICAL_SEGMENTERS.md).

```bash
python src/scripts/generate_busi_partitions.py --config BUSI_epito --model medsam --device cuda \
  --max-regions 254 --generate_at_least_n_masks 50 --save-partitions sam compact --compact-strategy greedy split --limit 1000
python src/scripts/generate_busi_partitions.py --config BUSI_epito --model medicosam --device cuda \
  --max-regions 254 --generate_at_least_n_masks 50 --save-partitions sam compact --compact-strategy greedy split --limit 1000
python src/scripts/view_partitions.py --config BUSI_epito --sam medsam --limit 5 --methods sam compact_greedy compact_split
```
