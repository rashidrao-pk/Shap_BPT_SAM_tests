# BPT versus SAM-guided BPT animations

Create one GIF per dataset using its first image with a completed compact partition:

```bash
python src/scripts/animate_bpt_hierarchies.py
```

This covers the configured MS-COCO, ImageNet, CelebA-HQ, CAMO, BUSI, PartImageNet, FunnyBirds, PACO, Kvasir-SEG, ISIC2018, DOTA and RefCOCO datasets. Missing inputs are reported; the other datasets continue. No detector, classifier, SAM inference, GPU or XAI budget is needed. The compiled `shap_bpt` extension must work in the active Python environment.

Each GIF shows **Input | SAM guidance | BPT | SAM-guided BPT**. Both trees use the XAI pipeline's actual pixel-level BPT builder. The animation moves from 256 regions down to one at matching region counts, omitting intermediate merges. This is a sequence of tree cuts, not wall-clock timing; the SAM-guided tree does not start with SAM regions as its leaves.

Choose a dataset and image, or a SAM variant:

```bash
python src/scripts/animate_bpt_hierarchies.py \
  --config MSCOCO_epito --sam sam2 --image-id 113867

python src/scripts/animate_bpt_hierarchies.py \
  --config ImageNet_epito BUSI_epito CAMO_epito --limit 2
```

The script reads `output.partitions`, `data.mask_dir_final` and `data.compact_strategy` from each config, including the new `compact_split` filenames. `--sam sam1` or `--sam sam2` selects the corresponding per-model folder. To use an older strategy folder:

```bash
python src/scripts/animate_bpt_hierarchies.py --config MSCOCO_epito \
  --partition-folder MSCOCO_sam2_large_254_compact-sam_200_split \
  --partition-type compact --image-id 113867
```

Omit the override when using the current layout: one folder with separate compact strategy files. `--partition-type compact_greedy` selects greedy, with legacy `_compact.npy` fallback; `--partition-type compact_split` selects split.

Outputs go under:

```text
<output.evaluations>/bpt_animations/<config>/<partition-folder>/<stage>/<image-id>.gif
```

Alongside each GIF, `.png` shows the first frame and `.json` records input paths, dimensions, guide parameters, region counts and tree construction times. Saved partitions are never modified. Colours help track regions; they do not indicate semantic classes or attribution values. Borders remain visible and ticks are hidden.

Adjust playback with `--frames 64 --fps 6 --start-regions 128`. The default uses full-resolution inputs. For a faster preview, add `--max-side 384`; this resizes the image and labels **before building both trees**, so it creates different trees and records the resized dimensions. Guide options such as `--guide-alpha` and `--label-words` match the XAI runners. Repeating a command replaces its rendered GIF, PNG and JSON.
