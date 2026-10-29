# Shared output paths

Each dataset config has three independent output roots:

```yaml
data:
  mask_dir_final: MSCOCO_sam2_large_62_refined_200
output:
  partitions: /beegfs/home/mrashid/datasets/SAM_BPT
  explanations: /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/outputs/results
  evaluations: /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/outputs/evaluations
```

| Key | Contents |
| --- | --- |
| `output.partitions` | Generated dataset/model partition folders and matching logs |
| `output.explanations` | Per-image explanations, predictions, GT, and resumable run files |
| `output.evaluations` | Aggregate metric CSVs, plots, HTML reports, progress/comparison reports, and partition previews |

`data.mask_dir_final` selects the partition subfolder for explanations/viewing.
Generation still derives its subfolder from dataset/model/budget arguments.
No additional dataset YAML is needed just to select different output purposes.
Relative roots resolve from the repository root. Explicit CLI output paths override
the applicable root.

For black background and budget 500, OD_yolo uses:

```text
<output.partitions>/MSCOCO_sam2_large_62_refined_200/
<output.explanations>/OD_yolo/MSCOCO_sam2_large_62_refined_200/bg_black_B500/139/
<output.evaluations>/OD_yolo/MSCOCO_sam2_large_62_refined_200/bg_black_B500/xai_results_report.html
<output.evaluations>/previews/MSCOCO_sam2_large_62_refined_200/000000000139.png
```

The explanation-side CSV remains a resume cache; a copy is published to the
evaluation folder as images finish. Aggregate evaluations do not move or duplicate
the original per-image images. Reports link across the two roots.

Older `data.masks_path` + `data.mask_dir` and `output.dir` + `output.folder`
configs remain readable through a shared compatibility helper. New keys take
precedence. Existing files are not moved. Resync the scripts and configs together
before starting a new EPITO process; an already-running job keeps its loaded paths.

## Commands

```bash
python src/scripts/view_partitions.py --config MSCOCO_epito --image-id 139
python src/scripts/check_partition_progress.py --limit 1000
python src/scripts/explanations/OD_yolo.py --config MSCOCO_epito --bg-type black --max-evals 500 --limit 2 --dry-run
python src/scripts/check_experiment_progress.py --config MSCOCO_epito
python src/scripts/summarize_results.py --config MSCOCO_epito --bg-type black --max-evals 500
python src/scripts/compare_experiments.py --config MSCOCO_epito
```

The summary command reads the selected explanation run and writes rebuilt reports
under the corresponding evaluation directory. `--results-dir` selects a different
input run explicitly. The comparison/progress commands discover nested runs from
`run_info.json`, including the new `bg_*_B*` folder layout.
