# Conceptual workflow

For runnable commands, see [SAM_Exps.md](SAM_Exps.md).
The diagram below describes the intended workflow; output names vary by script.

```md
Image / Image + Text
│
▼
┌──────────────────────────────┐
│ PARTITION GENERATION │
│ │
│ SAM1 ───── image only │
│ SAM2 ───── image only │
│ SAM3 ───── image + text │
└──────────────┬───────────────┘
│
▼
Unified partition format
image_id_sam1.npy
image_id_sam2.npy
image_id_sam3.npy
image_id_metadata.json
│
▼
┌──────────────────────────────┐
│ XAI │
│ BPT │
│ BPT + SAM guidance │
│ SAM + Kernel SHAP │
└──────────────┬───────────────┘
│
Classification /
Detection / VQA
│
▼
AUC+ / AUC- / IoU
```
