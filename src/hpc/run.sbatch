#!/usr/bin/env bash

#SBATCH --job-name=XAI-ShapBPT
#SBATCH --partition=epito
#SBATCH --reservation=XAI-ShapBPT
#SBATCH --nodelist=epito01
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/XAI-ShapBPT_%j.out
#SBATCH --error=logs/XAI-ShapBPT_%j.err


set -euo pipefail

echo "=========================================="
echo "XAI-ShapBPT job"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Started:       $(date)"
echo "=========================================="

# Project


# Environment
source /beegfs/home/mrashid/pt_312_sam/bin/activate

export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages
export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
export PYTHONUNBUFFERED=1

echo
echo "Python:"
which python
python --version

echo
echo "GPU:"
nvidia-smi

echo
echo "Starting training..."
echo

python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 200 \
  --limit 500


echo
echo "=========================================="
echo "COCO Partitions finished: $(date)"
echo "=========================================="
