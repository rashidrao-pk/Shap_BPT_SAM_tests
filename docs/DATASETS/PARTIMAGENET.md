## Download PartImageNet dataset

On a node with internet access, from the repository root:

```bash
python src/scripts/download_partimagenet.py --assets dataset
```

This downloads and validates `PartImageNet_Seg.zip`, extracts under `/beegfs/home/mrashid/datasets/PartImageNet`, and downloads `checkpoints/partimagenet/resnet50-11ad3fa6.pth` with a SHA256-prefix check. The [official archive](https://huggingface.co/datasets/turkeyju/PartImageNet/tree/main) is about 3.12 GB compressed; allow additional space for extraction and generated partitions.

The download script uses Python's standard library: it does not import NumPy, PyTorch, Kagglehub, or Xet. Existing downloaded files are reused. Invalid ZIPs produce an error before extraction; replace damaged archives with a fresh download.
