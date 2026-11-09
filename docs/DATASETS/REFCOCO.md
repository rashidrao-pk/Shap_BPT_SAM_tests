# RefCOCO

[RefCOCO](https://github.com/lichengunc/refer) pairs COCO photos with referring expressions: short phrases that pick out one object, such as "man on right" or "monitor far right". Each expression points at one COCO instance mask. This adds a vision–language task: CLIP scores how well the photo matches the phrase, and the explanation shows which regions drive that score.

| | |
| --- | --- |
| Split | UNC val: 1,500 images, 3,811 referred objects, 10,834 expressions |
| Images | COCO `train2014`, fetched one by one (only the 1,500 val images) |
| Ground truth | The COCO instance mask of the referred object |
| Model | [CLIP ViT-B/32](../MODELS/CLIP.md) |
| Download | `refcoco.zip`, 54 MB (annotations); images about 250 MB |
| Licence | Annotation code and files from [lichengunc/refer](https://github.com/lichengunc/refer) (Apache-2.0 repository; no separate data terms stated). COCO images under their Flickr licences, COCO annotations CC BY 4.0 |

## 1. Download the dataset

```bash
python src/scripts/download_datasets.py --dataset refcoco --assets all
```

1. **Annotations.** The authors' UNC server is offline, so the script fetches the archived original of `refcoco.zip` (archive.org, captured 2022-04-13, 53,993,993 bytes) and checks its SHA-256. It holds `instances.json` (COCO masks) and `refs(unc).p`.
2. **Safe reading.** `refs(unc).p` is a Python pickle. It is read with a loader that refuses every class or function, so only plain data loads and no code in the file can run.
3. **Expressions.** Every val expression is written to `RefCOCO/prepared/val/expressions.json`, with its image, object and category.
4. **Images.** Only the 1,500 val images are fetched from `images.cocodataset.org/train2014`, into `RefCOCO/prepared/val/images`. Repeat the command to retry failures.
5. **Model.** CLIP ViT-B/32 goes to `checkpoints/clip_vit_b32`.

Only `--split val` is supported.

## How records are built

- **One record per referring expression**, with the ID `<image>__ref<ref_id>_s<sent_id>`, for example `COCO_train2014_000000000716__ref49946_s142041`.
- **One expression per object by default** (its first sentence), so each referred object is explained once: 3,811 records. `--all-sentences` explains all 10,834.
- **Partitions are per image** and shared by that image's expressions.
- `--limit N` takes one object from each of N images first, so a short run still covers N different photos.

## 2. Models

- [CLIP ViT-B/32](../MODELS/CLIP.md)

## 3. Next steps

1. [Generate partitions](../Partitions/DATASETS/REFCOCO.md)
2. [Evaluate partitions against GT](../Partitions/EVALUATE_PARTITIONS.md)
3. [Run explanations](../XAI/DATASETS/REFCOCO.md)

## Limits on what you can claim

- **CLIP is not a referring-expression model.** It scores how well the whole photo matches the phrase. With two men in a photo, "man on right" may score high because of either man; the explanation shows what drove the score, not whether CLIP found the right one.
- **Location words.** RefCOCO expressions often say "left" or "right", which CLIP handles poorly. Explanations of such phrases are about what CLIP does, not about the phrase's meaning.
- **Possible leakage.** CLIP's training set (400 million web image–text pairs) is not public, and COCO photos come from Flickr, so the photos may have been seen in training.
- **Expressions per image are not independent.** The 3,811 objects come from 1,500 images; report both numbers.
- **The ground truth is one object.** Other objects of the same kind are not GT, even when CLIP's score depends on them.
- Research diagnostics only.
