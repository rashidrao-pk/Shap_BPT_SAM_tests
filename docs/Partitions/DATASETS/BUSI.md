# BUSI partitions on EPITO

First time: [download the dataset, classifier, and SAM checkpoints](BUSI_SETUP.md). The commands below are for routine generation after setup.

Set `data.image_dir` and `data.ground_truth.root` in `src/configs/BUSI_epito.yaml` to the folder containing `benign/`, `malignant/`, and `normal/`. Use the extracted path from the setup guide; it may include `Dataset_BUSI_with_GT`.

Annotation files (`*_mask.png`, `*_mask_1.png`, etc.) are excluded from SAM inputs. Numbered lesion masks are combined for evaluation. Grayscale PNG intensities are repeated across RGB channels without contrast enhancement.

After activating your SAM environment, test two images:

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_busi_partitions.py \
  --config BUSI_epito --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda --max-regions 62 --generate_at_least_n_masks 50 \
  --save-partitions sam compact refined --limit 2
```

Remove `--limit 2` for all images. Set `data.mask_dir_final` to the folder printed by the generator before running XAI.

Next: [SigLIP checkpoint](../MODELS/SigLIP_BUSI.md) and [BUSI explanations](../../XAI/BUSI.md).
