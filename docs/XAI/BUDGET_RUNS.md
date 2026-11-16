# Run XAI at chosen budgets, one after another

After a budget study, run the full XAI at the chosen budgets. For MS COCO these are **3000 and 5000** (see [budget steps](BUDGET_STEPS.md)). `run_budgets.py` runs the same runner command once per budget, in order, and can be stopped and restarted at any time.

## What runs at each budget

With `--explain_all`, the budget applies only to the three partition methods. Everything else keeps its own setting:

| Method | Run at 3000 | Run at 5000 |
| --- | --- | --- |
| BPT, BPT+SAM_c, SAM_c+KernelSHAP | 3000 | 5000 |
| LIME (`--lime-budgets`) | 1000 | 1000 |
| Shap-AA (`--aa-budget`) | 1000 | 1000 |
| GradCAM, IDG, GradExp, GradShap, LRP | no budget | no budget |

Each budget writes its own folder, `.../bg_full_B3000/` and `.../bg_full_B5000/`, so the two runs never mix.

## Run on Slurm (recommended)

From the repository root on the login node:

```bash
mkdir -p logs
sbatch --export=ALL,CHAIN_LEFT=40 src/hpc/run_xai_budgets.sbatch --budgets 3000 5000 -- \
  src/scripts/explanations/OD_yolo.py --config MSCOCO_epito_sam1 --model yolo11s \
  --show-gt --explain_all --no-heatmaps --limit 1000
```

- Everything after `--` is the runner command, **without** `--budget`; the driver adds it.
- A batch job keeps running when your terminal or the login node disconnects.
- **Time limit:** each job has 6 hours. Two minutes before the end, it records where it stopped. `CHAIN_LEFT=40` makes each job queue its own follow-up, up to 40 more jobs, which continues from there. When every budget is done, the driver cancels the waiting follow-up.
- **Errors:** if a budget fails twice in a row with an error (not a time limit), the chain stops so it doesn't waste allocations. Read the log, fix the error, and submit again.
- Without `CHAIN_LEFT`, one 6-hour job runs; submit the same command again to continue.

Watch it:

```bash
squeue -u mrashid
tail -f logs/XAI-budgets_<JOBID>.out
python src/scripts/explanations/run_budgets.py --dry-run --budgets 3000 5000 -- \
  src/scripts/explanations/OD_yolo.py --config MSCOCO_epito_sam1 --model yolo11s \
  --show-gt --explain_all --no-heatmaps --limit 1000
```

`--dry-run` prints each budget's status (pending, started, done), its attempts, and the time used so far. The runner command must be identical to the submitted one, because the state folder is chosen from it.

To stop: `scancel <JOBID>` for the running job and its queued follow-up (shown in the log as "Follow-up job ... queued").

## Run interactively

On a GPU node, inside `tmux` so a disconnect does not kill it:

```bash
python src/scripts/explanations/run_budgets.py --budgets 3000 5000 -- \
  src/scripts/explanations/OD_yolo.py --config MSCOCO_epito_sam1 --model yolo11s \
  --show-gt --explain_all --no-heatmaps --limit 1000
```

If it stops for any reason, run the same command again.

## How resuming works

There are two levels:

1. **Budgets.** A budget is marked done (`B<budget>.done`) only when the runner finishes all its images and exits cleanly. Done budgets are skipped. Budgets run in the order given; 5000 starts only after 3000 is done.
2. **Images.** Inside a budget, the runner skips images that are already complete. An image that was cut off halfway is redone from its start. See [resume](RESUME_EPITO.md).

State is kept in `results/budget_runs/<runner>_<hash>/`:

| File | Contents |
| --- | --- |
| `plan.json` | The runner command and budgets |
| `B<budget>_attempts.json` | Each attempt: job ID, node, start, end, seconds, outcome (`done`, `failed`, `interrupted`) |
| `B<budget>.done` | Written when the budget finishes, with its total time over all attempts |

A changed runner command (another config, `--limit`, or option) gets a new state folder. Changing the command does not change which images the runner treats as complete; the runner checks that itself.

## Time

From the MS COCO budget study (YOLO11s, full background, 19 images), one partition method takes about **3.7 min per image at 3000** and **6.1 min at 5000**. With three partition methods, that is about 11 and 18 minutes per image, before LIME, Shap-AA and the gradient methods. For 1000 images that is roughly 190 and 300 GPU-hours. Run `--limit 5` first and check the runner's timing summary before committing to 1000 images.

To halve the wall time, the two budgets can also run at the same time as two separate jobs, because they write separate folders: submit once with `--budgets 3000` and once with `--budgets 5000`. Never run two jobs with the same budget and command at once.
