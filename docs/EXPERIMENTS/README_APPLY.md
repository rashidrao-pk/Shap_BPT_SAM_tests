# Patch: pretrained BUSI + SAM/BPT attribution

This ZIP contains **only new files** for `Shap_BPT_SAM_tests`, preserving the existing repository.

## Apply on EPITO

From the project root, copy/extract the ZIP using its existing directory structure, e.g.:

```bash
cd /beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests
unzip -o busi_shapbpt_patch.zip
python -m pytest -q tests/test_busi_tree_attribution.py
```

See `docs/BUSI_PRETRAINED_SHAPBPT.md` for installation and end-to-end commands.

The included YAML is an experiment manifest; the script currently uses CLI flags for its configuration (YAML parsing is not wired in yet).

## Source-control workflow

```bash
git switch -c feature/busi-pretrained-shapbpt
git add src/scripts/run_busi_shapbpt.py src/configs/BUSI_epito.yaml docs/BUSI_PRETRAINED_SHAPBPT.md tests/test_busi_tree_attribution.py
git commit -m "Add pretrained BUSI SAM-BPT explainability pipeline"
git push -u origin feature/busi-pretrained-shapbpt
```

**Do not add datasets, model weights, or `results/` to Git.**

The connected GitHub application can read the repository but GitHub refused its branch creation request with HTTP 403. Therefore this patch is not yet present on any GitHub branch.
