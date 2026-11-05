#!/usr/bin/env bash
#SBATCH --job-name=AUC-Budget-ablation
#SBATCH --partition=epito
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --chdir=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
#SBATCH --output=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/ablation_study_%j.out
#SBATCH --error=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/logs/ablation_study_%j.err
set -euo pipefail
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
export PYTHONUNBUFFERED=1
exec python "$1/driver.py" "$1/manifest.json"
