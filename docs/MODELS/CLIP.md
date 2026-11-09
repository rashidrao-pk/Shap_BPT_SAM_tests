# CLIP ViT-B/32

| | |
| --- | --- |
| Used with | [RefCOCO](../DATASETS/REFCOCO.md) |
| Task | Image–text similarity |
| Runner | `VL_RefCOCO.py` |
| Checkpoint | `checkpoints/clip_vit_b32` (605 MB), `openai/clip-vit-base-patch32` at commit `3d74acf9` |

## Download

```bash
python src/scripts/download_datasets.py --dataset refcoco --assets model
```

Eight files are fetched from that pinned commit, each checked against its SHA-256: the weights (`pytorch_model.bin`), the config, and the tokenizer and preprocessing files. The repository has no `safetensors` copy; `transformers` loads the `.bin` file with `weights_only`, which does not run code from it.

This is the model the [FIxLIP demo](https://shapiq.readthedocs.io/en/latest/demos/fixlip/index.html) explains, so the score and model match that work.

## How it is scored

- The score is the **cosine similarity** between the image embedding and the embedding of the record's referring expression, as in FIxLIP. It usually lies between about 0.1 and 0.35. The text is fixed; only the image is explained.
- Text is embedded once per expression and cached.
- **Preprocessing.** CLIP normally resizes the short side and crops the centre square, which can cut off an object at the edge. Here the whole image is resized to 224×224, so every object stays visible. A 4:3 photo is squeezed by up to a third along its long side.

## Methods

- BPT, BPT+SAM_c and SAM_c+KernelSHAP; with `--explain_all`, LIME, Shap-AA, IDG, GradExp, GradShap and GradCAM.
- GradCAM uses the last vision block's pre-attention layer norm, on the 7×7 patch grid.
- LRP has no transformer backend and is reported as unsupported.

## Limits

- CLIP matches the whole photo to the phrase; it does not locate the referred object. See [RefCOCO](../DATASETS/REFCOCO.md#limits-on-what-you-can-claim).
- Its training data is not public, so overlap with COCO photos cannot be ruled out.
