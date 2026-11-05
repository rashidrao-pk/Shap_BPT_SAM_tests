# Resume XAI experiments on EPITO

## What is saved

| Runner | Resume |
| --- | --- |
| OC_ResNet, OC_ViT, COD_ViT, OD_DETR, OC_CelebA, OC_BUSI, OC_PartImageNet | Enabled by default; skips completed images with matching settings and input files |
| OD_yolo | Enabled by default; checks saved per-image outputs and CSV rows; all-method runs also check their completion marker/settings |
| run_busi_shapbpt.py run | Enabled by default; uses per-image completion markers and validates settings/input files |

Resume is **between images**. All requested explanations, AUC/evaluations, and outputs for an image must finish before it is marked complete. An interruption during an image repeats that image's XAI on restart; completed images are kept. There is no checkpoint inside a BPT sampling loop, Kernel SHAP, LIME, gradient method, or AUC calculation. Completed methods inside an unfinished image are not individually resumed. If one image itself takes longer than the job time limit, repeated six-hour allocations cannot finish it without reducing its work or obtaining a longer allocation.

For the shared runners, `<image_id>_complete.json` is written last. Missing artifacts mean the image needs processing again. The separate BUSI pipeline uses `<case_folder>/complete.json`; old results without this marker are recomputed once. It reuses saved SAM partitions and trees even when XAI must repeat.

## Continue an experiment

Repeat the **same command**, keeping the same config, checkpoint, budget, background, GT, AUC, and output options. **Remove `--no-resume`** from earlier testing commands:

```bash
python src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only \
  --explain_all --budget 1000 --no-heatmaps
```

The startup message reports `Pending` and `complete`. Shared runners reject incompatible settings rather than silently reuse results. YOLO's legacy mode has less comprehensive settings validation; use a different output directory for a different experiment. Do not run concurrent jobs writing the same experiment directory.

`--limit 10` means the first ten selected inputs, including already completed images. It does not request ten additional pending images. Increase the limit or omit it in shared runners to process more. `--image-ids` can select explicit batches without invalidating completed-image signatures.

The separate BUSI command defaults to two images. To select all currently downloaded BUSI cases:

```bash
python src/scripts/run_busi_shapbpt.py run \
  --limit 780 --sam sam1 sam2 --regions 62 --budget 504 --batch-size 8
```

Rerun it unchanged to continue. Reusing an existing experiment requires unchanged input/checkpoint file paths and modification times. For a new BUSI experiment use a new `--results` directory. `--no-resume` forces XAI recomputation but still reuses existing SAM partitions/trees; use a fresh directory when SAM settings change.

Changing from `--no-auc` to AUC enabled, adding `--explain_all`, or changing plotting/GT flags is not a separate cached stage: shared runners treat these as changed settings. There is currently no supported “explain now, evaluate later” command that reuses all saved methods automatically.

## Submit a separate six-hour job for each experiment

A Slurm batch job continues after the login terminal disconnects. Its allocation still ends at the requested wall time; submitting again resumes the application from completed images. See [Slurm sbatch](https://slurm.schedmd.com/sbatch.html).

From the repository root on the login node:

```bash
mkdir -p logs
sbatch src/hpc/run_xai.sbatch src/scripts/explanations/OC_ResNet.py \
  --config ImageNet_epito --show-gt --gt-only \
  --explain_all --budget 1000 --no-heatmaps
```

Submit other experiments independently, for example:

```bash
sbatch src/hpc/run_xai.sbatch src/scripts/explanations/OC_ViT.py \
  --config ImageNet_epito --show-gt --gt-only --explain_all --no-heatmaps

sbatch src/hpc/run_xai.sbatch src/scripts/explanations/COD_ViT.py \
  --config CAMO_epito --show-gt --explain_all --no-heatmaps

sbatch src/hpc/run_xai.sbatch src/scripts/explanations/OD_DETR.py \
  --config MSCOCO_epito --show-gt --explain_all --no-heatmaps

sbatch src/hpc/run_xai.sbatch src/scripts/explanations/OD_yolo.py \
  --config MSCOCO_epito --show-gt --explain_all --no-heatmaps

sbatch src/hpc/run_xai.sbatch src/scripts/run_busi_shapbpt.py run \
  --limit 780 --sam sam1 sam2 --regions 62 --budget 504 --batch-size 8

sbatch src/hpc/run_xai.sbatch src/scripts/explanations/OC_PartImageNet.py \
  --config PartImageNet_epito --show-gt --explain_all --no-heatmaps \
  --checkpoint checkpoints/partimagenet/resnet50-11ad3fa6.pth --weights IMAGENET1K_V2
```

The launcher uses the existing EPITO partition/GPU request, a six-hour time limit, and `pt_312_sam`. It does not request a reservation or pin a node. Adjust scheduler resources if your allocation requires different settings. It does not automatically resubmit jobs. Model/SAM checkpoints must be available on the compute node; BUSI defaults can be overridden with `--sam1-checkpoint` and `--sam2-checkpoint`.

After a job ends, repeat its identical `sbatch` command for another allocation. No checkpoint files need copying. `tmux` or `nohup` alone does not extend a Slurm allocation past its wall time.

Monitor jobs and logs:

```bash
squeue -u mrashid
tail -f logs/XAI_JOBID.out logs/XAI_JOBID.err
python src/scripts/check_experiment_progress.py --results-root results/explanations
```

Replace `JOBID` with the submitted job ID. The progress scanner covers the shared/YOLO experiment layout. The separate BUSI pipeline writes under `results/busi` and reports its resume counts on startup; its markers are not indexed by that scanner.
