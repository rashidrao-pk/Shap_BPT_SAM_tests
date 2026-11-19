# MedSAM and MedicoSAM

Medical SAM variants that generate partitions for BUSI, ISIC 2018 and Kvasir-SEG. Like SAM-1 and SAM-2, they are
not the models being explained.

| | MedSAM | MedicoSAM |
| --- | --- | --- |
| Used by | Partitions for BUSI, ISIC 2018, Kvasir-SEG (`--model medsam`) | Same (`--model medicosam`) |
| Architecture | SAM ViT-B, fine-tuned on medical images | SAM ViT-B, fine-tuned on medical images |
| Package | none extra: loads into `segment_anything` (SAM-1) | same |
| Checkpoint | `checkpoints/medsam/medsam_vit_b.pth` | `checkpoints/medicosam/vit_b_medicosam.pt` |
| Masks from | a box grid with box refinement (MedSAM needs box prompts) | SAM's automatic mask generator |

## Download

On EPITO (internet access), from the repository root:

```bash
# MedSAM: official Google Drive folder from the MedSAM README
python -m pip install gdown
mkdir -p checkpoints/medsam
gdown --folder https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN -O checkpoints/medsam

# MedicoSAM: direct link from its README
mkdir -p checkpoints/medicosam
wget https://owncloud.gwdg.de/index.php/s/f5Ol4FrjPQWfjUF/download -O checkpoints/medicosam/vit_b_medicosam.pt

ls -lh checkpoints/medsam checkpoints/medicosam
```

- Each file should be at least about 350 MB. A file of a few KB is an HTML error page: download it again.
- If the MedSAM file has another name (older releases: `medsam_20230423_vit_b_0.0.1.pth`), rename it or pass
  `--checkpoint PATH`.
- If Google refuses `gdown` (quota or sign-in page), download in a browser and copy it with `scp` to
  `checkpoints/medsam/`.

## Check and use

Load test, generation commands, how the box grid works and caveats:
[Medical segmenters](../../Partitions/SAM/MEDICAL_SEGMENTERS.md).
