# Setup

Everything you need before generating partitions and running explanations, in the order to do it. Each step links
to the page with the commands.

| Step | What | Page |
| --- | --- | --- |
| 1 | Python environment on each machine (EPITO ARM, x86 GPU nodes) | [Environment](#1-environment) |
| 2 | ShapBPT, this package (Cython extension) | [ShapBPT](#2-shapbpt) |
| 3 | Segmenters for the partitions: SAM-1, SAM-2, MedSAM, MedicoSAM | [Segmenters](#3-segmenters-for-the-partitions) |
| 4 | Datasets | [Datasets](DATASETS/README.md) |
| 5 | Models that are explained, and their checkpoints | [Models](MODELS/README.md) |
| 6 | Packages for every XAI method | [XAI methods](XAI_METHODS.md) |
| 7 | Check that everything is in place | [Check](#7-check-everything) |

Then: [generate partitions](../Partitions/README.md) and [run XAI](../XAI/README.md).

## 1. Environment

| Machine | CPU | Environment | Setup |
| --- | --- | --- | --- |
| EPITO (A100), gracehopper (GH200) | aarch64 | `/beegfs/home/mrashid/pt_312_sam` | [EPITO setup](../Partitions/full/EPITO_SETUP.md) |
| H200, L40S, V100S, T4, L4 | x86_64 | `/beegfs/home/mrashid/pt_312_sam_x86` | `bash src/hpc/setup_x86_env.sh` on the node; [other GPU nodes](../Partitions/full/OTHER_GPU_NODES.md) |

On any node, `source src/hpc/node_env.sh` picks the right environment for the CPU type, builds it if missing,
activates it and checks torch, the GPU, the BPT extension and SAM. From the login node,
`bash src/hpc/gpu.sh v100` (or `a100`, `h100`, `l40s`, `best`, ...) gets a GPU and does the same in one command:
see [GPU sessions](../Partitions/full/GPU.md).

The x86 environment pins the official CUDA 12.6 PyTorch (2.7.1 / torchvision 0.22.1), because the node driver
supports CUDA ≤ 12.9; `torch-constraints.txt` in the environment stops other packages from replacing it.

## 2. ShapBPT

ShapBPT (version in `shap_bpt/_version.py`, currently 1.1) has a Cython extension, `shap_bpt/bpt.pyx`, which is
compiled once per CPU type. Both builds live side by side in the repository
(`bpt.cpython-312-aarch64-linux-gnu.so`, `bpt.cpython-312-x86_64-linux-gnu.so`).

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
python -m pip install Cython
python -m pip install --no-deps --no-build-isolation --force-reinstall -e .
python -c "import shap_bpt, pprint; pprint.pprint(shap_bpt.build_info())"
```

- Run it on EPITO for the ARM environment and on an x86 **GPU node** for the x86 one. The login node has no
  Python headers (`Python.h`), so compiling there fails.
- `build_info()` should show version 1.1, the `.so` for this CPU and `extension_older_than_source: False`.
  `True` means `bpt.pyx` changed after the build: rebuild.
- After a change to `bpt.pyx`, delete the old `.so` for that CPU first
  (`rm -f shap_bpt/bpt.cpython-*-$(uname -m)-linux-gnu.so`).

## 3. Segmenters for the partitions

| Segmenter | Package | Checkpoint | Page |
| --- | --- | --- | --- |
| SAM-1 ViT-B | `segment_anything` (GitHub) | `checkpoints/sam1/sam_vit_b_01ec64.pth` | [SAM-1](MODELS/SAM1.md) |
| SAM-2.1 Large (default), Tiny | `sam2` (editable checkout in `$SAM2_DIR`) | `$SAM2_DIR/checkpoints/sam2.1_hiera_large.pt` | [SAM-2](MODELS/SAM2.md) |
| MedSAM, MedicoSAM (BUSI, ISIC 2018, Kvasir-SEG) | none extra: load into SAM-1 ViT-B | `checkpoints/medsam/medsam_vit_b.pth`, `checkpoints/medicosam/vit_b_medicosam.pt` | [MedSAM and MedicoSAM](MODELS/MEDICAL_SAM.md) |

## 4. Datasets

[Datasets](DATASETS/README.md) lists each dataset, its ground truth and its model. Most are downloaded by one
script:

```bash
python src/scripts/download_datasets.py --dataset funnybirds --assets dataset   # paco, kvasirseg, isic2018, dota, refcoco, all
```

MS-COCO, ImageNet, CAMO, CelebA-HQ, BUSI and PartImageNet have their own download steps on their pages.

## 5. Models and checkpoints

[Models](MODELS/README.md) has one page per explained model with its download command. For the six datasets of the
download script, `--assets model` fetches the checkpoint (with a SHA-256 check):

```bash
python src/scripts/download_datasets.py --dataset all --assets model
```

torchvision models (ResNet-50, VGG16, ViT-B/16) and DETR download themselves into the torch / Hugging Face cache
on first use; on a node without internet, run once on EPITO first.

## 6. XAI methods

[XAI methods](XAI_METHODS.md): which package each method needs, the install command, and an import check.

## 7. Check everything

```bash
python src/scripts/summary_setup.py                     # Datasets, checkpoints, packages, partitions: READY or what is missing
python src/scripts/summary_setup.py --config BUSI_epito  # One dataset
python src/scripts/summary_setup.py --strict             # Exit 1 unless everything is READY for XAI
```

Missing items name the page in this folder that fixes them. More options: [Check your setup](DATASETS/README.md#check-your-setup).
