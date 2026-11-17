# Label bitsets and the guide penalty (BPT+SAM_c)

BPT+SAM_c builds the BPT with a SAM partition as a guide. Each cluster carries the set L of partition labels it contains. A candidate merge's normal colour/shape priority D is multiplied by a penalty that grows with |L|, the number of labels the merged cluster would contain. Clusters with the lowest priority merge first. The code is in `shap_bpt/bpt.pyx`.

| Setting | Builder argument | Runner flag | Default (v1) |
| --- | --- | --- | --- |
| Label words W (labels < 64·W) | `label_words` = 1, 2, 4 or `'auto'` | `--label-words` | 1 |
| Penalty exponent α | `guide_alpha` ≥ 0, or `'hard'` | `--guide-alpha` | 2 |
| Penalty offset ε | `guide_eps` > 0 | `--guide-eps` | 1 |
| Count label 0 (pixels without a SAM mask) in L | `count_unassigned` | `--no-count-unassigned` turns it off | on |

Priority of a merge: **D × (|L| + ε)^α**. The defaults reproduce the v1 builder exactly, (popcount + 1)², which `tests/golden/bpt_trees_v1.npz` checks bit for bit.

- **α = 0:** the penalty is 1, so the tree equals the unguided BPT (Proposition 2(i)).
- **α = `'hard'`:** the α → ∞ limit, implemented in lexicographic order rather than as a large α, which would overflow. The priority is |L|·10⁶ + log(1 + D), so every merge inside one label happens before any merge across labels. Every 8-connected component of every label then becomes exactly one tree node (Proposition 2(ii)); `shap_bpt.tree_contains_region(tree, mask)` checks this.
- **δ and Δ:** the builder records the smallest and largest unguided priority over all evaluated merges, as `min_base_priority` and `max_base_priority`.

## Label maps

- **Any integer type works.** The label map is converted to int64 before bits are set. v1 shifted in the map's own type, so a `uint8` map lost every label ≥ 8, and `uint16` every label ≥ 16. The runners were safe, because their loader already returned int64.
- **Rejected with `ValueError`, never wrapped:** labels ≥ 64·W, negative labels, or a shape that differs from the image. `'auto'` uses ⌈(max label + 1) / 64⌉ words, at most 4 (labels < 256).
- **The runner's partition cap follows W:** at most 64·W − 1 labels, counting label 0 (63 by default, 127 for W = 2, 255 for W = 4 or `auto`).

## Runner outputs

With non-default settings:
- **Folder:** the run folder gets a suffix, for example `bg_full_B3000_W2_a4_e1` (plus `_u0` when label 0 is not counted).
- **Method name:** with `--explain_all`, the guided method is named for example `BPT+SAM_c@a4W2-3000`, listing only the settings that differ.
- **Method status:** the per-method status JSON records the full settings under `guide`.

The defaults leave names, folders and resume signatures exactly as before.

## Partitions with more than 62 regions

The batch generators do not keep SAM's raw masks, so 126- or 254-region partitions need a new SAM run:

```bash
sbatch --export=ALL,CHAIN_LEFT=10 src/hpc/run_partitions.sbatch src/scripts/generate_coco_partitions.py \
  --config MSCOCO_epito --model sam1 --checkpoint "$PROJECT_DIR/checkpoints/sam1/sam_vit_b_01ec64.pth" \
  --sam1-variant vit_b --device cuda --max-regions 126 --generate_at_least_n_masks 200 --save-partitions sam compact
```

This writes `MSCOCO_sam1_vit_b_126_compact-sam_200`. The audit labels it `SAM1-VB-K126`. Explain from it with `--label-words 2` (126) or `--label-words 4` (254); with the default W = 1 the loader caps it back to 63 labels.

## Build time

`src/scripts/analysis/bpt_build_time.py` times unguided and guided builds for W = 1, 2, 4 and writes `results/paper/T1/build_time.csv`. On 20 MS-COCO images, on a laptop CPU, with older partitions of up to about 190 labels, the median build took 0.61 s unguided, and 0.67, 0.68 and 0.69 s guided at W = 1, 2 and 4.

## Rebuild after pulling

`bpt.pyx` changed, and compiled files are not in git. Rebuild once in each environment:

```bash
python setup.py build_ext --inplace
```
