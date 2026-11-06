# Datasets

Setup for each dataset, including download commands and the pretrained checkpoint to explain.

| Dataset | Ground truth for XAI | Frozen classifier | Were the explained images in training? | Runner |
| --- | --- | --- | --- | --- |
| [FunnyBirds](FUNNY_BIRD.md) | Class-defining parts | Authors' ResNet-50 | No: test split | `OC_FunnyBirds` |
| [PACO](PACO.md) | Object parts in scenes | ImageNet ResNet-50 | Not via COCO; ImageNet not audited | `OC_PACO` |
| [Kvasir-SEG](KVASIR_SEG.md) | Polyp masks | Kvasir v2 ViT-B/16 | **Likely yes** | `OC_KvasirSEG` |
| [ISIC 2018](ISIC_2018.md) | Lesion masks | Task 3 EfficientNet-B0 | No shared image IDs | `OC_ISIC2018` |
| [PartImageNet](PARTIMAGENET.md) | Object parts | ImageNet ResNet-50 | Possibly | `OC_PartImageNet` |
| [BUSI](BUSI.md) | Lesion masks | BUSI SigLIP | **Yes**, see its model card | `OC_BUSI` |

Models trained on the explained images are still valid to explain, since an attribution describes the model. They do not support claims about held-out performance, so state the overlap when reporting results.

## Quick start

From the repository root, on a node with internet access:

```bash
python src/scripts/download_datasets.py --dataset funnybirds   # or paco, kvasirseg, isic2018, all
```

Then generate partitions and run the explanation commands in that dataset's page. Check progress with:

```bash
python src/scripts/check_partition_progress.py --config FunnyBirds_epito PACO_epito KvasirSEG_epito ISIC2018_epito
```

All four use the existing SAM1 and SAM2 checkpoints for partitions. Paths in the `*_epito.yaml` configs are for EPITO.

To include them in the budget study, name them explicitly:

```bash
python src/hpc/prepare_full_budget_study.py --repo . --study <folder> --datasets FunnyBirds PACO KvasirSEG ISIC2018
```

The default study remains the original six datasets.

## Added datasets

- [x] MS-COCO
- [x] ImageNet
- [x] PartNet
- [x] Celeb-A-HQ
- [x] CAMO
- [x] BUSI
- [x] FunnyBirds
- [x] PACO: Parts and Attributes of Common Objects
- [x] Kvasir-SEG: Segmented Polyp Dataset for Computer Aided Gastrointestinal Disease Detection (https://datasets.simula.no/kvasir-seg)
- [x] ISIC-2018
