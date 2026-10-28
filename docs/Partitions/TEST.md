```bash
python src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito \
  --model sam2 \
  --checkpoint "$SAM2_DIR/checkpoints/sam2.1_hiera_tiny.pt" \
  --device cuda \
  --max-regions 62 \
  --points-per-side 32  \
  --points-per-batch 64  \
  --pred-iou-thresh 0.6  \
  --  \
  --  \
  --  \
  --  \



