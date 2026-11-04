# BUSI pretrained model + SAM/BPT hierarchical explanations

This add-on is **inference-only**: no model fitting, fine-tuning, or weight updates.

## Sources

- Dataset: [BUSI Kaggle mirror](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset), originally Al-Dhabyani et al., _Data in Brief_ (2020), DOI 10.1016/j.dib.2019.104863. 780 source images (437 benign, 210 malignant, 133 normal), plus lesion mask PNGs. Public mirror carries its own reuse terms.
- Frozen classifier: [ghostsas001/busi-breast-ultrasound-siglip](https://huggingface.co/ghostsas001/busi-breast-ultrasound-siglip), a three-class SigLIP model. Its model card documents the exact `Resize((224,224))`, `ToTensor()`, and `Normalize([0.5]*3,[0.5]*3)` preprocessing replicated here. The model card warns that duplicates may cross image-level train/test splits.

**Research-only** results are not clinically validated and cannot establish independent held-out accuracy on BUSI because this pretrained model was trained on BUSI. The test reports the labels and predictions but does not claim unbiased generalization. If evaluating accuracy, obtain a truly independent external cohort or the model's exact held-out IDs and audit duplicates.

## EPITO setup

Activate an allocated NVIDIA A100 compute node (not `mercurio-login`) and the known-good venv:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
source /beegfs/home/mrashid/pt_312_sam/bin/activate
unset PYTHONPATH
python -m pip install 'transformers>=4.57,<5' 'huggingface_hub>=0.34,<3' kagglehub 'safetensors>=0.4' PyYAML
python -c 'import numpy,torch,torchvision,transformers; print(numpy.__version__, torch.__version__, torch.cuda.is_available())'
```

Avoid any pip operation that upgrades/replaces the EPITO custom ARM64 PyTorch/torchvision. For extra safety, create a separate ARM64 venv using `--system-site-packages` and test paths before installing the transformers dependencies. If `pip` proposes replacing torch, cancel the installation and use `--no-deps` selectively while checking dependent imports.

## Download dataset and checkpoint

```bash
python src/scripts/run_busi_shapbpt.py download
python src/scripts/run_busi_shapbpt.py inventory
```

The download saves the BUSI images to `/beegfs/home/mrashid/datasets/BUSI` (using Kagglehub's `output_dir`) and the frozen pretrained Hugging Face model to `checkpoints/busi_siglip`. Datasets and checkpoints are **not** committed to Git.

## Run two ultrasound images through all stages

```bash
python src/scripts/run_busi_shapbpt.py run --limit 2 --sam sam1 sam2 --regions 62 --budget 504 --batch-size 8
```

The script invokes existing `generate_sam_partitions.py` and `build_sam_bpt.py`, then estimates the contribution of each leaf using random child-order permutations through the verified binary tree. It saves `attribution.npy`, `explanation.png`, `metrics.json`, and insertion/deletion curves for both SAM variants.

- SAM1 checkpoint: `checkpoints/sam1/sam_vit_b_01ec64.pth`.
- SAM2.1 Tiny checkpoint: `checkpoints/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt` (existing symlink expected). Override via `--sam2-checkpoint` if necessary.
- `--budget 504` for 62 leaves evaluates 8 complete permutations × 63 classifier calls = 504 calls **per method per image**, plus 42 insertion/deletion evaluations and the original score.
- Attributions are **hierarchically constrained random-permutation marginal estimates**. Verify equivalence to the specific original ShapBPT/Owen estimator before reporting these as exact Owen values in a publication.
- Localization uses positive attributions: point-in-mask and equal-lesion-area top-ranked IoU. Normal images have no lesion mask and report null localization.
- Approximate model AUC uses probabilities; attributions use target logit scores. Insertion/deletion ranks whole leaves and uses a 21-step curve.
- Avoid using ground-truth lesion masks in SAM partition generation or hierarchy construction.

## Results

`results/busi/<class>_<image_stem>/sam1|sam2/{attribution.npy,explanation.png,tree.json,...}` and `results/busi/summary.json`.

To scale up use `--limit 10` etc. Expect significant time for SAM inference and repeated SigLIP calls on EPITO. No measured runtimes or accuracy are claimed until this has actually been executed on the cluster.

## Methodological caveats

1. The separate model's BUSI training data may overlap evaluation data; document this limitation.
2. A leaf from SAM may include disconnected areas if refinement merges them; the tree script has a documented non-adjacent fallback for disconnected graphs.
3. Grayscale scans are represented as RGB triplets for SAM and the SigLIP classifier.
4. The chosen classifier output is a **classification** decision, not a diagnosis; explanations do not establish clinical relevance.
5. For formal Owen-value benchmarking, connect the exact published estimator from the original ShapBPT implementation. This package provides an explicit, auditable tree-consistent Monte Carlo estimator.

## Download fails while importing NumPy

`download` and `inventory` do not need NumPy, Pillow, or PyTorch from this script. Update `src/scripts/run_busi_shapbpt.py` to the version with imports inside the computation functions, then rerun the download command.

A missing `numpy.core._multiarray_umath` indicates a separate compiled NumPy installation problem; the final source-directory message does not establish that you are running from a NumPy source tree. See [NumPy troubleshooting](https://numpy.org/doc/stable/user/troubleshooting-importerror.html).

Before repairing the environment, check architecture on the login node and allocated GPU node:

```bash
uname -m
python -c 'import sys, platform; print(sys.executable); print(platform.machine()); print(sys.version)'
```

If their CPU architectures differ, install binary packages and run inference on the compute architecture for which the environment is intended. Do not replace ARM packages with login-node x86 wheels in the same environment.

On the matching compute node, reinstall only NumPy using that environment's interpreter:

```bash
unset PYTHONPATH
python -m pip install --force-reinstall --no-cache-dir --only-binary=:all: --no-deps 'numpy==1.26.4'
python -c 'import numpy; print(numpy.__version__, numpy.__file__)'
```

This repairs NumPy only; it does not reconcile other packages' version requirements. Run `python -m pip check` afterward to identify any remaining conflicts.
