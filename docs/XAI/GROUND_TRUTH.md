# Ground-truth overlays

`--show_gt` (alias `--show-gt`) is available in the explanation runners and the
partition viewer. It displays GT without changing model scores, explanations,
or the existing AUC/faithfulness calculations.

## Dataset config

The input masks are defined under `data.ground_truth`; output roots stay in the
existing `output` section. Your EPITO paths are included in the dataset YAMLs.

| Dataset | Format | Matching |
| --- | --- | --- |
| CAMO | `binary` | Same filename stem under `GT/Train` (or configured split) |
| CelebAMask-HQ | `celebamask_hq` | Numeric image ID → `<id // 2000>/<id:05d>_<part>.png` |
| ImageNetS | `imagenets` | Same image stem under the configured variant's segmentation folders |
| MS-COCO | Existing COCO annotations | Class name → category ID → instance segmentation |

Example ImageNet settings:

```yaml
data:
  ground_truth:
    format: imagenets
    root: /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/data/data_full/ImageNet/ImageNetS
    variant: 50
```

Set `variant` to the downloaded split (`50`, `300`, or `919`). A root pointing
directly to the matching `validation-segmentation` folder also works. Multiple
matching files are rejected rather than guessed. ImageNetS does not provide a
mask for every original ImageNet validation image; unavailable annotations are
saved/displayed with `annotation not available` status.

## Which class is shown?

- **COCO explanations:** GT for the class being explained, as before.
- **CAMO:** binary camouflaged-object foreground. This is dataset GT, not an ImageNet class-specific mask.
- **CelebA previews:** union of available annotated parts, or choose `--gt-class hair`, `skin`, `l_eye`, etc.
- **ImageNet explanations:** predicted-class GT via the classifier index → synset → ImageNetS label mapping.
- **ImageNet previews:** all annotated foreground by default, because there is no model prediction in the viewer.

For ImageNetS, `--gt-class foreground` shows every annotated object;
`--gt-class n07930864` selects a synset; a number selects an **ImageNetS label ID**,
not a zero-based classifier index. Background `0` and ignored pixels `1000` are
excluded. RGB masks are decoded as `R + 256 * G`, following the
[official ImageNet-S format](https://github.com/LUSSeg/ImageNet-S#qa).

For default torchvision ImageNet models, the loader can use timm's packaged
ImageNet synset list without importing timm into the viewer. If unavailable,
configure `class_index_file` under `ground_truth` as a zero-indexed JSON mapping
(e.g. `{"0": ["n01440764", "tench"], ...}`) or a synset-per-line text file.
Custom classifiers need an explicit mapping or `--gt-class foreground`.

The ImageNetS `categories_file` can point to the official variant's synset list.
It is also discovered automatically below the GT root. For a single labelled
category in a synset-named directory, that directory supplies the mapping when
the full list is absent. Mapping gaps produce an explicit empty mask/status;
they never silently reuse a classifier index as a segmentation ID. Official
919-class synset merges are handled.

## Alignment and files

Masks are matched to the original image stem and resized to the image dimensions
with **nearest-neighbor interpolation** when needed. Metadata records the source
shape, paths, whether resizing occurred, selected class/part, and GT scope.
CelebA's grouping and part-mask layout follow the
[official preprocessing code](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py).

New GT loaders save:

```text
<image-id>_gt.npy          # Selected boolean mask
<image-id>_gt_labels.npz   # Compressed original semantic/part labels, aligned to the image
<image-id>_gt.png          # Mask and overlay
<image-id>_gt.json         # Sources, class scope, alignment, and status
```

The combined XAI figure/HTML report includes the overlay. `--no-heatmaps` skips
individual explanation PNGs but keeps GT and the combined plot. The original
YOLO COCO helper retains its existing GT file layout.

To add GT to an already completed explanation run, use `--no-resume` or a separate
`--output-dir` when the settings check requests it. Partition regeneration is not
needed. CelebA currently has partition previews; no CelebA attribute classifier
was supplied in the earlier runner requests.

## Tests

Synthetic tests cover binary masks, part groups/padding, RGB IDs above 255,
background/ignore labels, predicted-class mapping, unavailable masks, ambiguous
filenames, nearest-neighbor resizing, previews, and CAMO explanation GT artifacts.
Actual EPITO annotation files are not accessible from the local workspace.

## Reference ImageNet notebook

The EPITO config uses ImageNetS50, matching
[N1_1_Run_experiments_testing.ipynb](https://github.com/rashidrao-pk/shap_bpt_tests/blob/main/notebooks/E1_E2_E3_E7/N1_1_Run_experiments_testing.ipynb).
The loader finds the nested `ILSVRC2012_img_val_S/ImageNetS50/validation-segmentation`
layout below the configured root. When providing `categories_file`, use
`ImageNetS_categories_im50.txt`: label `i + 1` maps to line `i` in its original
order. `class_index_file` accepts the notebook's `imagenet_class_index.json`.
Both files may live outside the dataset when their paths are configured.

The notebook decodes RGB as `R + 256 * G`, excludes 1000, and uses nearest-neighbor
mask resizing. The runner follows those rules, aligning GT to the explanation
image dimensions rather than forcing every image to 224 × 224.

The notebook's experiment loop keeps only single-category images whose annotation
class matches the predicted class. `--show-gt` does not apply that experiment
filter: it shows the selected class mask and explicitly reports missing or
unmatched GT. It does not silently discard an explanation.
