# EPITO: CAMO Train partitions

Use the existing SAM GPU environment from [EPITO.md](EPITO.md#1-start-a-gpu-session).
Sync `generate_camo_partitions.py` and `src/configs/CAMO_epito.yaml` to EPITO,
along with the shared generation scripts if they are not current.

The config scans the folder containing your example image:

```text
/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests/data/data_full/CAMO-V.1.0/Images/Train
```

Only Train images are included. Original filename stems such as
`camourflage_00001` are preserved. No detection annotations are needed.

## Check paths

```bash
python src/scripts/generate_camo_partitions.py --limit 2 --dry-run
```

## Test two images

```bash
export SAM2_DIR=/beegfs/home/mrashid/repos/third_party/sam2
python src/scripts/generate_camo_partitions.py \
  --config CAMO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --device cuda \
  --max-regions 62 \
  --generate_at_least_n_masks 50 \
  --save-partitions sam compact boosted filled boosted_filled refined \
  --limit 2
```

The script defaults to SAM2 Tiny and target 50. It resolves the checkpoint from
`SAM2_DIR`, or the standard EPITO checkout above when that variable is absent.
The explicit arguments above make the settings visible and can override defaults.

## Generate the complete Train split

Remove `--limit 2` from the command. Completed images with matching settings are
skipped on restart. Progress appears as text tqdm. Outputs and matching log:

```text
/beegfs/home/mrashid/datasets/SAM_BPT/CAMO_sam2_tiny_62_refined_50/
/beegfs/home/mrashid/datasets/SAM_BPT/CAMO_sam2_tiny_62_refined_50.log
```

The target is 50 **raw masks**, with bounded threshold retries; reaching it is
not guaranteed. `sam` and `refined` maps are capped at 62 regions; notebook stages
keep their original behavior. Choose stages with `--save-partitions`; without
it only `sam` and `refined` are saved. Each results JSON lists saved paths,
counts, timings, and tuning history. See [stage details](EPITO.md#choose-which-partition-stages-to-save).

## Preview

```bash
python src/scripts/view_partitions.py \
  --config CAMO_epito \
  --image-id camourflage_00001 \
  --methods sam compact boosted filled boosted_filled refined
```

Use only stages you saved. Replace `--image-id camourflage_00001` with `--limit 5`
to preview the first five completed images. Preview images and statistics JSONs
are saved under the configured output directory in
`previews/CAMO_sam2_tiny_62_refined_50/`. Viewing needs no GPU.

These are generic region partitions, not predicted camouflage class labels or
CAMO detection/segmentation evaluation results.
