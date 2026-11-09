# Leaderboard: which XAI method wins

## After each run

Every explanation run ends by ranking its methods. Nothing extra is needed:

```bash
python src/scripts/explanations/OC_CelebA.py \
  --config CelebAHQ_epito --attribute Brown_Hair \
  --show-gt --limit 1 --explain_all --no-heatmaps --no-auc --no-resume
```

The verdict is printed at the end and saved in the run's evaluation folder:

| File | Content |
| --- | --- |
| `winner.md` | Overall top 3, and per metric: winner, runner-up, and whether the lead is clear |
| `leaderboard.csv` | Per method and metric: mean, median, mean rank, share of images won; plus `overall_rank` |
| `metric_winners.csv` | Per metric: winner, Friedman p-value, methods the winner can't be separated from |
| `method_boxplots.png` | One panel per metric, one box per method, the winner in blue |
| `method_ranks.png` | Mean rank of every method on every metric |

To rebuild it for a finished run, including an `OD_yolo` run:

```bash
python src/scripts/explanations/xai_leaderboard.py dataset \
  results/evaluations/COD_ViT/CAMO_sam2_tiny_62_compact-sam_50/bg_full_B1000
```

## Across datasets and domains

```bash
python src/scripts/explanations/xai_leaderboard.py all --budget 1000 --bg-type full
```

It reads every `results/evaluations/<runner>/<partitions>/bg_<type>_B<budget>/auc_results_all_images_current_run.csv` and writes to `results/evaluations/xai_leaderboard/`:

| File | Content |
| --- | --- |
| `winner.md` | Final ranking, datasets and domains won, Friedman test, winner per domain |
| `overall.csv` | Per method: overall rank, datasets won, domains won, mean rank per metric |
| `dataset_ranks.csv`, `dataset_ranks.png` | Overall rank of each method on each dataset |
| `domain_ranks.csv`, `domain_ranks.png` | The same per domain |
| `rank_boxplots.png` | Per metric, the spread of each method's rank over datasets |
| `runs.csv`, `run_ranks.csv` | Which runs were used or left out, and their ranks |

Always pass `--budget` and `--bg-type`, so one dataset isn't counted at two settings. `--methods` picks the methods to compare; by default, those found in at least half of the runs. Runs missing one of them are left out and listed.

## How the winner is decided

- **Ranks, not raw values.** Methods are ranked within each image (1 = best), so an easy image or a confident model doesn't outweigh the others. Only images where every method has a value are used.
- **Direction.** Higher is better for insertion AUC, ROAD gap and LeRF, best IoU, pointing game, sensitivity-n and stability; lower is better for deletion AUC, ROAD MoRF and time.
- **Overall rank** is the mean of the insertion AUC, deletion AUC and ROAD gap ranks, whichever were computed. Localization, robustness and time are reported beside it, not mixed in.
- **ROAD** uses the scores relative to the unmodified image when every image has them, and raw scores otherwise (as with CelebA logits). Ranks within an image are the same either way.
- **Clear lead.** Within a run, the winner of a metric is tested against every other method with a paired Wilcoxon test, Holm-corrected. A lead is clear when p < 0.05 against all of them. With fewer than 10 images, the verdict is marked as a preview.
- **Across datasets,** each dataset's mean ranks are averaged, with every dataset counted once; several runs of one dataset (another model or partitions) are averaged first. The Friedman test over datasets says whether the methods differ at all (Demšar, JMLR 2006).

Domains: natural images (ImageNet, MS-COCO), object parts (PartImageNet, PACO), synthetic (FunnyBirds), camouflage (CAMO), faces (CelebA-HQ), medical (BUSI, Kvasir-SEG, ISIC 2018).
