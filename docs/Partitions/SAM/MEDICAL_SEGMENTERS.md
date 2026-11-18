# Medical segmentation priors: MedSAM and MedicoSAM

On the medical datasets (BUSI, ISIC 2018, Kvasir-SEG), the generic SAM ViT-B prior often fails the partition audit. Two SAM ViT-B models fine-tuned on medical images are added as alternative priors. Both plug into the existing partition generators and produce the usual `<id>_sam.npy`, `<id>_compact_greedy.npy` (and any other `--compact-strategy`) and `<id>_results.json`, so explanations, previews and the audit work unchanged.

| Prior | `--model` | Trained with | How it runs here | Folder |
| --- | --- | --- | --- | --- |
| MedSAM ([bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM), Apache-2.0) | `medsam` | box prompts only | **box grid** (default); `--medsam-prompt point-grid` as an out-of-distribution ablation | `<Dataset>_medsam_vitb_254_compact-sam_50_greedy` (`..._vitb-pointgrid_...` for the ablation) |
| MedicoSAM ([computational-cell-analytics/medico-sam](https://github.com/computational-cell-analytics/medico-sam), MIT) | `medicosam` | point and box prompts | SAM's point-grid automatic mask generator with MedicoSAM weights | `<Dataset>_medicosam_vitb_254_compact-sam_50_greedy` |

Code: `src/scripts/medical_segmenters.py`, wired into `generate_sam_partitions.build_generator`, so every `generate_*_partitions.py` accepts `--model medsam | medicosam`. Tests: `tests/test_medical_segmenters.py`.

## 1. One-time setup on EPITO

Work in an EPITO session, which has the ARM environment and internet access:

```bash
srun -p epito --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
```

No new model package is needed: both checkpoints load into the `segment_anything` ViT-B that is already installed. Only the two weight files are downloaded.

**MedSAM.** The official checkpoint is in the Google Drive folder linked from the MedSAM README. `gdown` downloads Drive folders from the command line:

```bash
python -m pip install gdown
mkdir -p checkpoints/medsam
gdown --folder https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN -O checkpoints/medsam
ls -lh checkpoints/medsam
```

The file expected by default is `checkpoints/medsam/medsam_vit_b.pth`. If the folder holds a file with another name (older releases used `medsam_20230423_vit_b_0.0.1.pth`), either rename it or pass `--checkpoint PATH`. If `gdown` is refused by Google (quota or sign-in page), download the file in a browser and copy it:

```bash
scp medsam_vit_b.pth epito-mercurio:/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/checkpoints/medsam/
```

**MedicoSAM.** The README gives a direct link:

```bash
mkdir -p checkpoints/medicosam
wget https://owncloud.gwdg.de/index.php/s/f5Ol4FrjPQWfjUF/download -O checkpoints/medicosam/vit_b_medicosam.pt
ls -lh checkpoints/medicosam
```

Each file should be at least about 350 MB, the size of the ViT-B weights; a training checkpoint can be larger. A file of a few KB is an HTML error page and must be downloaded again.

**Record the checksums** for the paper's reproducibility statement. Each image's metadata also stores them:

```bash
sha256sum checkpoints/medsam/medsam_vit_b.pth checkpoints/medicosam/vit_b_medicosam.pt
```

## 2. Check that both models work

```bash
python -m pytest -q -rs tests/test_medical_segmenters.py
```

All 8 tests should pass, none skipped. The last two load the real weights and must find a bright synthetic disk with IoU > 0.5. A skip names the missing checkpoint path.

- **`load_state_dict` reports missing or unexpected keys:** the file is not a SAM ViT-B checkpoint. Check the download.
- **The MedicoSAM loader falls back to a full pickle load:** this happens when the file is a training checkpoint holding more than the weights. `segmenter.state_dict.full_pickle` in the metadata records it.

**If the real-model tests fail**, find out where with the diagnostic. It compares the two models with SAM-1 ViT-B, which segments the synthetic disk with IoU 0.998:

```bash
python src/scripts/diagnose_medical_segmenters.py --device cuda
python src/scripts/diagnose_medical_segmenters.py --device cuda \
  --image "/beegfs/home/mrashid/datasets/BUSI/benign/benign (1).png" \
  --mask "/beegfs/home/mrashid/datasets/BUSI/benign/benign (1)_mask.png"
```

For each model it reports:
- a direct prompt on the target (one box, plus one point except for MedSAM) with no filtering;
- the automatic generator with filters off and at the default thresholds;
- the spread of predicted IoU and stability.

How to read it:
- **Good direct prompt, but the automatic run loses the target:** the filters are the problem (for MedSAM, likely its untrained IoU head).
- **Bad direct prompt:** the problem lies in the model, its preprocessing or the test image.

## 3. Generate partitions

The settings match the existing SAM runs on these datasets: up to 254 regions, a mask target of 50, and the `sam` and `compact` stages. The commands run in the EPITO session; for long runs, use the chained sbatch job below.

```bash
COMMON=(--device cuda --max-regions 254 --generate_at_least_n_masks 50 --save-partitions sam compact)

# BUSI: all images (437 benign, 210 malignant, 133 normal; only benign and malignant have lesion masks)
python src/scripts/generate_busi_partitions.py --config BUSI_epito --model medsam "${COMMON[@]}" --limit 1000
python src/scripts/generate_busi_partitions.py --config BUSI_epito --model medicosam "${COMMON[@]}" --limit 1000

# ISIC 2018 and Kvasir-SEG: 1000 images
python src/scripts/generate_isic2018_partitions.py --config ISIC2018_epito --model medsam "${COMMON[@]}" --limit 1000
python src/scripts/generate_isic2018_partitions.py --config ISIC2018_epito --model medicosam "${COMMON[@]}" --limit 1000
python src/scripts/generate_kvasirseg_partitions.py --config KvasirSEG_epito --model medsam "${COMMON[@]}" --limit 1000
python src/scripts/generate_kvasirseg_partitions.py --config KvasirSEG_epito --model medicosam "${COMMON[@]}" --limit 1000

# Ablation: MedSAM with SAM's point grid (out of distribution), BUSI only
python src/scripts/generate_busi_partitions.py --config BUSI_epito --model medsam --medsam-prompt point-grid "${COMMON[@]}" --limit 1000
```

Start with `--image-id` or `--limit 5` to see a few results first. Runs resume per image, as for SAM.

**As chained 6-hour jobs**, from the login node:

```bash
sbatch --export=ALL,CHAIN_LEFT=5 src/hpc/run_partitions.sbatch src/scripts/generate_busi_partitions.py \
  --config BUSI_epito --model medsam --device cuda --max-regions 254 --generate_at_least_n_masks 50 \
  --save-partitions sam compact --limit 1000
```

## 4. Look at them and audit them

The three medical configs know the new folders (`data.mask_dir_final_medsam`, `data.mask_dir_final_medicosam`), so the viewer selects them with `--sam`:

```bash
python src/scripts/view_partitions.py --config BUSI_epito --sam medsam --limit 5 --methods sam compact
python src/scripts/view_partitions.py --config BUSI_epito --sam medicosam --limit 5 --methods sam compact
```

Side by side with SAM, scored against the lesion masks:

```bash
python src/scripts/compare_partition_folders.py --config BUSI_epito --name medical_priors --limit 50 \
  --folders BUSI_sam1_vit_b_254_compact-sam_50 BUSI_sam2_large_254_compact-sam_50 \
            BUSI_medsam_vitb_254_compact-sam_50 BUSI_medicosam_vitb_254_compact-sam_50 \
  --labels SAM1-VB SAM2-L MedSAM MedicoSAM
```

The audit with a verdict per prior is what fills the paper's quality table. Its rows are labelled `MedSAM-VB`, `MedicoSAM-VB` and `MedSAM-VB-PG`:

```bash
python src/scripts/evaluate_partitions.py --config BUSI_epito ISIC2018_epito KvasirSEG_epito --all-sam \
  --paper-dir results/paper/T2
```

To run XAI on a medical prior, set `data.mask_dir_final` in the config to its folder.

## How it works

**MedSAM box grid** (`MedSAMBoxGridGenerator`). MedSAM only ever saw box prompts in training, so prompting it with SAM's 32 × 32 point grid would test it out of distribution. Instead, each image is covered by square boxes at three scales:
- **Box sizes:** side = 1/2, 1/4 and 1/8 of the shorter image side.
- **Positions:** stride = half the side, plus a last row and column flush with the image border. A square image gets 9 + 49 + 225 = 283 boxes.

Processing follows the official `MedSAM_Inference.py`:
1. Resize the image to 1024 × 1024 (bicubic, anti-aliased) and min-max normalise it to [0, 1].
2. Run the image encoder once per image.
3. Scale the boxes to the 1024 frame and decode them in batches with `multimask_output=False`.
4. Take the sigmoid, upsample bilinearly to the image size, and threshold at 0.5.

Each box gives one mask. Filtering and NMS then work as in SAM's own generator:
- **Filtering:** a mask is kept if its predicted IoU and stability score pass the thresholds (0.6 by default). The stability score is computed as SAM computes it, on the logits ±1.
- **Duplicates:** box NMS at 0.7 removes them.
- **Mask target:** if fewer masks than the target survive, both thresholds are relaxed (0.6 → 0.4 → 0.2), exactly as for SAM. The decoder outputs are cached, so relaxing costs no extra model calls.

**MedicoSAM.** It was trained with points and boxes in micro_sam, which keeps SAM's preprocessing. Its weights are loaded into the official `SamAutomaticMaskGenerator`, so it runs exactly like SAM-1 ViT-B with the same settings, only with different weights. micro_sam is not needed.

**Grayscale.** BUSI images are grayscale. The generators open every image with `PIL.Image.convert('RGB')`, which copies the grey value to all three channels. That is the same as `src/medical/grayscale_adapter.to_rgb` and as MedSAM's own `np.repeat`. Intensities are not rescaled before the model's own preprocessing.

**Metadata.** Each `<id>_results.json` has a `segmenter` entry: model, backbone, checkpoint SHA-256, prompt mode, the code path that ran, whether the prompts are in distribution, and, for MedSAM, the box scales. `time_sam` holds the time per image, as for SAM.

## Caveats

- **MedSAM's IoU head was not trained.** MedSAM was fine-tuned with mask losses only, so its predicted IoU is SAM's original IoU head applied to fine-tuned features. It is still used for the threshold and as the weight in compact selection, for parity with SAM, and the stability score is applied too. Both values are saved per mask.
- **Box grid vs point grid.** The box grid is our choice of automatic prompts for MedSAM, not part of MedSAM. The point-grid ablation shows how much it matters.
- **Training-data overlap.** MedSAM's training collection includes public ultrasound, dermoscopy and endoscopy data. Check the overlap with BUSI, ISIC 2018 and Kvasir-SEG before claiming generalisation; the paper's appendix should disclose it.
- **Embedding check.** The preprocessing test reproduces the official script's steps line by line. A direct comparison of embeddings with the official script was not run, because that script is not importable as a function.
