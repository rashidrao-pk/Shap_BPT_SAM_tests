# FunnyBirds explanations

Run from the repository root on a GPU node. First [download the dataset](../../DATASETS/FUNNY_BIRD.md), get the checkpoint ([FunnyBirds ResNet-50](../../MODELS/FUNNYBIRDS_RESNET50.md)), and [generate partitions](../../Partitions/DATASETS/FUNNYBIRDS.md). Make sure `data.mask_dir_final` names the generated folder.

## Run

```bash
python src/scripts/explanations/OC_FunnyBirds.py \
  --config FunnyBirds_epito --show-gt --limit 1 \
  --budget 1000 --no-heatmaps --explain_all
```

The three partition methods share the budget: BPT, BPT+SAM_c, and SAM_c+KernelSHAP. `--explain_all` adds LIME, Shap-AA, LRP, GradCAM, IDG, GradExp, and GradShap, and needs `requirements-xai-all.txt`. Remove `--limit 2` for all completed partitions. Add `--dry-run` to check paths without loading the model.

The checkpoint path comes from `model.checkpoint` in the config. Use this runner, not `OC_ResNet`: the model takes 256×256 input without ImageNet normalization, and gets 7 of 40 sampled images right if preprocessed for ImageNet.

The explained class is the model's top prediction, fixed for each image. Classes are named `class_0` to `class_49`, matching the image's parent folder.

## Ground truth

`--show-gt` outlines the five class-defining parts: beak, eye, foot, tail, and wing. To select one, add `--gt-class beak`. `--gt-class body` also works, but every class shares the body, so it is not class evidence.

## Reading the results

- Check whether attribution falls on parts instead of the body or background distractors.
- The union of parts is an upper bound on the evidence: only parts that differ from competing classes separate them.
- These runs report insertion/deletion AUC and GT localization, not the paper's FunnyBirds protocols. Do not compare them with the paper's scores.

## Outputs and resume

```text
<output.explanations>/OC_FunnyBirds/<partition-folder>/bg_full_B1000/<image-id>/
<output.evaluations>/OC_FunnyBirds/<partition-folder>/bg_full_B1000/
```

Repeat the same command to resume: completed images are skipped. `--no-heatmaps` skips individual method images, and keeps the combined plots and numeric results.
