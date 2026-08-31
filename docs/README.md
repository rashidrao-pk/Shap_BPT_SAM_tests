# ShapBPT using SAM

## move Dataset to epito

```bash
# Upload MS COCO Dataset

zip /Users/rashid/data/PhD/datacloud_data/repos/XAI/ShapBPT-SAM/shap_bpt_sam/src/data/risultati_120

scp /Users/rashid/data/DS/MS_COCO.zip mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/datasets

# Upload SAM Results
scp /Users/rashid/data/PhD/datacloud_data/repos/XAI/ShapBPT-SAM/shap_bpt_sam/src/data/risultati_120.zip mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/datasets/SAM_BPT

```

## Unzip Uploaded Data:

```bash
unzip -o /beegfs/home/mrashid/datasets/SAM_BPT/risultati_120.zip \
  -d /beegfs/home/mrashid/datasets/SAM_BPT \
  -x "__MACOSX/*"

ls -lah /beegfs/home/mrashid/datasets/SAM_BPT/risultati_120/

du -sh /beegfs/home/mrashid/datasets/SAM_BPT/risultati_120/

find /beegfs/home/mrashid/datasets/SAM_BPT/risultati_120/120_refined_filled -type f | wc -l

```

- partition path
  - [/beegfs/home/mrashid/datasets/risultati_120](/beegfs/home/mrashid/datasets/risultati_120)
- Dataset Path
  - /beegfs/home/mrashid/datasets/MS_COCO
- Code Repo:
  - [github.com/ElleR01/shap_bpt_sam](www.github.com/ElleR01/shap_bpt_sam)

```bash
ls
```

## Copy Code

```bash
cd /beegfs/home/mrashid/repos/XAI
git clone git@github.com:ElleR01/shap_bpt_sam.git
```

```bash
# srun -p epito --gres=gpu:a100:1 --pty bash
# tmux
tmux new -s shapbpt

## GET PREVIOUS Session
tmux ls
ps -ef | grep tmux
echo $TMUX

# Detach
# Ctrl+b
# then d

#  ATTACH with previous
tmux attach -t shapbpt

tmux list-panes -t shapbpt -F '#{pane_pid} #{pane_current_command} #{pane_current_path}'


srun -p epito --gres=gpu:a100:1 --pty bash
source /beegfs/home/mrashid/pt_312/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
```

```bash
cd /beegfs/home/mrashid/repos/XAI/shap_bpt_sam
pip install Cython
pip install -r /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/requirements.txt
python setup.py build_ext --inplace
python setup.py install

## Verify ShapBPT installation
python -c "import shap_bpt; print('shap_bpt_version:', shap_bpt.__version__)"
```

```bash
python /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/src/scripts/check_setup.py
sinfo --format="%P %G %C"

export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/

#  TEST GPU
python src/scripts/test.py
python src/scripts/test_GPU.py
#
python /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/src/scripts/run_yolo_full.py

python -m py_compile src/scripts/run_yolo_full.py
python src/scripts/run_yolo_full.py --help

python src/scripts/run_yolo_full.py --limit 2 --max-evals 10

python src/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --max-evals 500 \
  --verbose-level "high" \
  --limit 2


python src/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --limit 2 \
  --max-evals 500 \
  --verbose-level medium \
  --eval-batch-size 2048 \
  --auc-batch-size 1024 \
  --model yolo26s


python src/scripts/run_yolo_full.py \
  --config MSCOCO_epito \
  --max-evals 500 \
  --verbose-level medium \
  --eval-batch-size 2048 \
  --auc-batch-size 1024 \
  --model yolo11s

```

```bash
# GENERATE RPORT
python src/scripts/summarize_results.py --config MSCOCO_epito --boxplots

# Inspect Outliers
python src/scripts/summarize_results.py \
  --config MSCOCO_epito \
  --outlier-report \
  --outlier-top-n 100

```

> Warning

```bash
# Flush Results?
# rm -r /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/src/results/xai_results
```

```bash
squeue -u mrashid
squeue -u mrashid -o "%.18i %.12P %.20j %.8T %.10M %.10l %.6D %R"

sattach <JOB_ID>.0

srun --jobid=<JOB_ID> --pty bash


scancel 92873

sacct -j 92623 \
  --format=JobID,JobName%25,State,Elapsed,ExitCode,MaxRSS,NodeList

scancel 92623

squeue -u mrashid

find /beegfs/home/mrashid/repos/XAI/shap_bpt_sam \
  -type f -mmin -180 \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' \
  | sort | tail -100

```

```bash
zip -r /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/results.zip /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/src/results/
scp mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/repos/XAI/shap_bpt_sam/results.zip .

zip -r /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/xai_results_E3_1.zip /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/src/results/xai_results_E3_1


cd /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/src/results
zip -r /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/xai_results_E3_1.zip xai_results_E3_1
zip -r /beegfs/home/mrashid/repos/XAI/shap_bpt_sam/results.zip results

scp mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/repos/XAI/shap_bpt_sam/xai_results_E3_1.zip ~/Downloads/
scp mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/repos/XAI/shap_bpt_sam/results.zip ~/Downloads/
```
