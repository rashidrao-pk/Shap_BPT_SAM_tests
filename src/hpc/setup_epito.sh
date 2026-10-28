srun -p epito --gres=gpu:a100:1 -J "ShapBPT" --time=06:00:00 --pty bash
source /beegfs/home/mrashid/pt_312_sam/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages
export PROJECT_DIR=/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
cd "$PROJECT_DIR"