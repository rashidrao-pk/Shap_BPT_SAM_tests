TODO plan

## TODO Plan

1. [x] Add XAI methods
   - [x] kernal-Shap with only SAM

- [ ] Check for available-Datasets/models suitable for ShapBPTSAM
- [x] Keep only 1 exp per model - E1, E2, E3, E4
- [x] Evaluation metrics: - keep AUC+-, Stability: Top-10 Jaccard only
- [x] Across Experiment analysis:
- [x] Normalize AUC numbers to keep comparable

## Partition Generation

Add SAM Variants

- [x] SAM-1
- [x] SAM-2
- [ ] SAM-3

## DATASETS

Add Datasets

- [x] MS-COCO
- [x] ImageNet
- [x] PartNet
- [x] Celeb-A-HQ
- [x] CAMO
- [x] BUSI
- [] FunnyBird dataset
- [] PACO: Parts and Attributes of Common Objects
- [] Seg-Polyp: Segmented Polyp Dataset for Computer Aided Gastrointestinal Disease Detection. (https://datasets.simula.no/kvasir-capsule-seg/)
- [] ISIC-2018

## Models

Add Models

- [x] Yolo
- [x] ViT-b
- [x] ResNet-50
- [x] VGG
- [x] CNN

### XAI part

- [x] Add more points (200) in AUC
- [x] Background Replacement Values as 5 (Black, White, Noise, Blurr)
- [x] Single Budget Selection based on Ablation — pilot/selection workflow implemented; run real-data EPITO pilots before choosing the minimum
