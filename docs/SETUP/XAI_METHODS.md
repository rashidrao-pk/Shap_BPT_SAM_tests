# XAI methods: what to install

Every method of the study and what it needs. Most are either part of this repository or come from three packages
(Captum, LIME, SHAP), installed with one command. Run everything inside the SAM environment
([Setup](README.md#1-environment)), on both the ARM and the x86 environment.

## Install

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
python -m pip install -r requirements-xai-all.txt                 # ARM (EPITO)
python -m pip install -c /beegfs/home/mrashid/pt_312_sam_x86/torch-constraints.txt -r requirements-xai-all.txt   # x86
python -m pip check
```

`requirements-xai-all.txt` pins versions that work with the SAM environment:

| Package | Version | Why pinned |
| --- | --- | --- |
| numpy | 1.26.4 | SAM and SAM3 code needs NumPy 1.x |
| captum | 0.8.0 | LRP, GradCAM, GradShap |
| lime | 0.2.0.1 | LIME |
| shap | 0.49.1 | GradExp (GradientExplainer) |
| scikit-image | ≥ 0.22, < 0.26 | LIME segmentation, BPT features |
| opencv-python | 4.11.0.86 | 4.12 needs NumPy 2 on Python 3.12 |

Do not let pip replace torch: on x86 always pass the constraints file; on EPITO, if pip wants to install a torch,
stop and check. If `pip check` reports Transformers with `huggingface-hub 2.x`, run
`python -m pip install 'huggingface-hub>=1.5,<2'`.

## Methods and their sources

| Method (name in results) | Family | Implementation | Needs |
| --- | --- | --- | --- |
| `BPT-<b>` | Owen values on a colour/shape BPT | `shap_bpt` (this repo) | ShapBPT built ([Setup §2](README.md#2-shapbpt)) |
| `BPT+SAM_c-<b>` | Owen values on the SAM-guided BPT (ours) | `shap_bpt` | ShapBPT + SAM partitions ([Partitions](../Partitions/README.md)) |
| `SAM_c+KernelSHAP-<b>` | KernelSHAP with SAM regions as players | `src/scripts/utils_xai.py` (native) | SAM partitions; no `shap` needed |
| `Shap-AA-<b>` | Owen values on axis-aligned splits | `shap_bpt` | ShapBPT |
| `LIME-<b>` | Local surrogate on superpixels | `lime` | `lime`, `scikit-image` |
| `LRP` | Relevance propagation (ε rule) | `captum.attr.LRP` | `captum`; CNNs (ResNet-50 style) |
| `GradCAM` | Class activation map | `captum.attr.LayerGradCam` | `captum` |
| `IDG` | Integrated Decision Gradients | `all_methods.py` (native) | torch only |
| `GradExp` | Expected gradients | `shap.GradientExplainer` | `shap` |
| `GradShap` | Gradient SHAP | `captum.attr.GradientShap` | `captum` |
| `hShap-<b>`, `hShap-CS-<b>` | Hierarchical Shapley on a quadtree (Teneggi et al. 2022) | `explanations/hshap_native.py` (native) | numpy only |
| `EAC-<b>` | Explain Any Concept, surrogate on SAM concepts (Sun et al. 2023) | `explanations/eac_native.py` (native) | torch + SAM partitions |

Gradient methods (LRP, GradCAM, IDG, GradExp, GradShap) need a differentiable PyTorch classifier. For detectors
(OD_yolo, OD_DETR) they are reported as unsupported, and LRP also for ViTs; unsupported methods get a status
entry and no map.
Which method runs on which architecture: [All methods](../XAI/ALL_METHODS.md#architecture-support).

How they are switched on:

```bash
# --explain_all adds LIME, Shap-AA, LRP, GradCAM, IDG, GradExp and GradShap;
# --extra-methods adds h-Shap, h-Shap with cycle spinning and EAC
python src/scripts/explanations/OC_ResNet.py --config ImageNet_epito --limit 2 \
  --explain_all --extra-methods hshap hshap-cs eac
```

Without `--explain_all`, a run computes only the three partition methods (BPT, BPT+SAM_c, SAM_c+KernelSHAP).

## Re-implemented baselines (no install)

- **h-Shap:** written against our masked predictor, so it uses the same backgrounds as every other method and works
  on detectors and CLIP. The authors' package (`h-shap`) is not installed; a comparison with it is still open.
- **EAC:** the original code (github.com/Jerry00917/samshap) has an academic-only licence and is not copied here. The
  native version follows the paper; the one deviation (an MLP surrogate instead of the classifier's frozen head) is
  recorded in each result's `info`.

Details and budgets: [All methods](../XAI/ALL_METHODS.md#extra-baselines-h-shap-and-eac---extra-methods).

## Models behind the methods

The methods explain the models in [Models](MODELS/README.md). Packages needed by those models:

| Package | For |
| --- | --- |
| `torchvision` | ResNet-50, VGG16, ViT-B/16, ISIC EfficientNet-B0 |
| `ultralytics` (`--no-deps`) | YOLO11s, YOLO11s-OBB |
| `transformers` | DETR, CLIP, BUSI SigLIP, Kvasir ViT |
| `timm` | DETR backbone |
| `safetensors` | BUSI SigLIP, ISIC EfficientNet-B0 |

```bash
python -m pip install transformers timm safetensors
python -m pip install --no-deps ultralytics
```

## Check

```bash
python - <<'PY'
import importlib
for name in ['numpy', 'torch', 'torchvision', 'shap_bpt', 'captum', 'lime', 'shap', 'skimage', 'cv2',
             'transformers', 'ultralytics', 'segment_anything', 'sam2']:
    try:
        module = importlib.import_module(name)
        print(f'{name:18s} {getattr(module, "__version__", "ok")}')
    except Exception as error:
        print(f'{name:18s} MISSING ({error.__class__.__name__}: {error})')
from shap_bpt import bpt
print('BPT extension     ', bpt.__file__)
PY
```

Then a two-image smoke run of every method:

```bash
python src/scripts/explanations/OC_ResNet.py --config ImageNet_epito --gt-only --limit 2 \
  --explain_all --extra-methods hshap hshap-cs eac --no-heatmaps --no-resume
```

`<image>_method_status.json` in each image folder says which methods ran, and why any was skipped.
