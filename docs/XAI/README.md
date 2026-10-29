# Run XAI experiments

All runners compare **BPT**, **BPT + SAM_c**, and **SAM_c + Kernel SHAP**.
SAM_c reads the saved `*_compact.npy` partition maps.

## Before running

Activate your EPITO GPU environment and work from the repository root. In the
dataset config, set `data.mask_dir_final` to the generated partition folder.
That folder must include compact maps and completed `*_results.json` files.

[Output paths](../OUTPUT_PATHS.md) come from the same config:

```yaml
output:
  partitions: /beegfs/home/mrashid/datasets/SAM_BPT
  explanations: /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/outputs/results
  evaluations: /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/outputs/evaluations
```

Append `--dry-run` to any command below to check paths and selected images without
loading model weights. Start with two images; remove `--limit 2` for all completed
inputs. Original ImageNet/CAMO filename stems are preserved.

## MS-COCO: object detection

YOLO:

```bash
python src/scripts/explanations/OD_yolo.py --config MSCOCO_epito --bg-type black --max-evals 50 --limit 2
```

DETR (default checkpoint: `facebook/detr-resnet-50`):

```bash
python src/scripts/explanations/OD_DETR.py --config MSCOCO_epito --bg-type black --max-evals 50 --limit 2
```

Add `--show_gt` to either detector for the COCO segmentation of the explained
class. DETR uses summed detection scores by default; `--class-aggregation max`
uses the maximum score per class. `--threshold` controls DETR score filtering
(default `0` for continuous masked-image scores); displayed boxes use at least
`0.3` confidence. DETR additionally requires `transformers` and `timm`.

## ImageNet: object classification

ResNet-50:

```bash
python src/scripts/explanations/OC_ResNet.py --config ImageNet_epito --bg-type black --max-evals 50 --limit 2
```

ViT-B/16:

```bash
python src/scripts/explanations/OC_ViT.py --config ImageNet_epito --bg-type black --max-evals 50 --limit 2
```

Both use torchvision's default pretrained ImageNet weights and preprocessing.
No COCO annotation file is needed. The explained score is the predicted class
probability; these runs measure model faithfulness, not ImageNet label accuracy.

## CAMO: ViT explanations

```bash
python src/scripts/explanations/COD_ViT.py --config CAMO_epito --bg-type black --max-evals 50 --limit 2
```

This uses the CAMO images/partitions and defaults to an ImageNet-trained ViT-B/16
classifier. For a CAMO-finetuned classifier, add `--checkpoint /path/to/model.pt`
and `--class-names /path/to/classes.json`. The checkpoint must match torchvision's
ViT-B/16 classifier architecture; spatial COD segmentation checkpoints require a
different model adapter. Class names may also be stored in checkpoint metadata.
The default score is ImageNet classification confidence on CAMO images.

## Show ground truth

GT paths are configured in `data.ground_truth` in each dataset YAML.
Add `--show_gt` to the explanation command:

```bash
python src/scripts/explanations/OC_ResNet.py --config ImageNet_epito --limit 2 --show_gt --no-heatmaps
python src/scripts/explanations/OC_ViT.py --config ImageNet_epito --limit 2 --show_gt --no-heatmaps
python src/scripts/explanations/COD_ViT.py --config CAMO_epito --limit 2 --show_gt --no-heatmaps
```

ImageNetS GT is filtered to the predicted ImageNet class when its synset mapping
is available. Use `--gt-class foreground` to display all annotated objects instead.
CAMO shows its binary camouflaged-object foreground, independently of the ViT's
ImageNet class prediction. Missing annotations/mappings are labelled explicitly.

CelebA face-part GT can be shown in partition previews:

```bash
python src/scripts/view_partitions.py --config CelebAHQ_epito --image-id 0 --methods compact --show_gt --gt-class hair
```

Omit `--gt-class hair` to combine available face-part masks. Partition previews
for ImageNet and CAMO also accept `--show_gt`. See [GT formats and mapping](GROUND_TRUTH.md).

## Outputs and useful options

Outputs follow the same layout for every runner:

```text
<output.explanations>/<runner>/<partition-folder>/bg_black_B50/<image-id>/
<output.evaluations>/<runner>/<partition-folder>/bg_black_B50/
```

Per-image outputs include predictions, three attribution maps (NPZ), optional
heatmaps/XAI grid, and metrics. Evaluation folders contain CSV summaries, plots,
and `xai_results_report.html`. Existing complete images are skipped by default.

- `--max-evals 500`: increase the BPT explanation budget.
- `--kernel-shap-samples 500`: set the Kernel SHAP coalition budget (defaults to `--max-evals`).
- `--model-batch-size 4`: lower model-forward memory for the four new runners.
- `--auc-num-samples 101`: insertion/deletion sampling for the new runners.
- `--no-auc`: skip metric evaluation for a quick explanation smoke run.
- `--no-heatmaps`: skip individual method heatmap PNGs; keep the combined XAI figure, evaluation plots, and numeric attribution NPZ. Supported by all five runners.
- `--no-plots`: save numeric outputs without explanation/evaluation plots.
- `--no-resume`: recompute selected images when changing settings; alternatively use a separate `--output-dir`.

Run settings and source-image/partition stamps are checked before reusing results.
A completion manifest is written only after all required artifacts succeed.
Pretrained weights download on first use if absent; DETR supports
`--local-files-only` when its checkpoint is already cached.

See [partition generation](../Partitions/README.md) and [YOLO details](OD_yolo.md).
CelebA-HQ currently has partition/preview tooling; an attribute-classification
explanation runner was not requested here.

## Local validation

The four new entry points were smoke-tested on CPU with cached pretrained weights,
real ShapBPT/Kernel SHAP, and a resized bundled photo. Partition maps in that smoke
run were four-quadrant fixtures, not SAM predictions. AUC/faithfulness and resume
are also covered by automated tests. These checks do not establish accuracy on
the full EPITO datasets. See [validation details](VALIDATION.md).

### Combined figure layout

All explanation runners save a three-row figure:

1. Original image, BPT partitions, SAM_c partitions.
2. BPT, SAM_c, and SAM_c_kernelshap attributions.
3. AUC+, AUC−, and ground truth when `--show-gt` is enabled.

The BPT panel cuts the explanation tree to the SAM_c region count. With
`--no-auc`, the curve panels say AUC disabled. `--no-heatmaps` still keeps
this combined figure. Use `--no-resume` to regenerate previously saved figures.
