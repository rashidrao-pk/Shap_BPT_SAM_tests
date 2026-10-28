## SAM-1

## SAM-2

```text
params = {
    "points_per_side": 32,
    "points_per_batch": 64,

    "pred_iou_thresh": 0.6,
    "stability_score_thresh": 0.6,
    "stability_score_offset": 1.0,

    "mask_threshold": 0.0,

    "box_nms_thresh": 0.7,
    "crop_n_layers": 0,
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 512 / 1500,
    "crop_n_points_downscale_factor": 1,

    "point_grids": None,
    "min_mask_region_area": 0,

    "output_mode": "binary_mask",
    
    "use_m2m": True,
    "multimask_output": True,
    }
```