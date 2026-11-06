## Download PartImageNet dataset

On a node with internet access, from the repository root:

```bash
python src/scripts/download_partimagenet.py --assets dataset
```

This downloads and validates `PartImageNet_Seg.zip` and extracts it under `/beegfs/home/mrashid/datasets/PartImageNet`. The [official archive](https://huggingface.co/datasets/turkeyju/PartImageNet/tree/main) is about 3.12 GB compressed; allow additional space for extraction and generated partitions.

The download script uses Python's standard library: it does not import NumPy, PyTorch, Kagglehub, or Xet. Existing downloaded files are reused. Invalid ZIPs produce an error before extraction; replace damaged archives with a fresh download.

## Models

- [ResNet-50](../MODELS/RESNET50.md)

## Next steps

1. [Generate partitions](../Partitions/DATASETS/PARTNET.md)
2. [Run explanations](../XAI/DATASETS/PARTIMAGENET.md)
