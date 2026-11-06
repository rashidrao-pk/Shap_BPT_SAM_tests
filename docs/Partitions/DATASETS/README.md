### DATASETS

- [BUSI](docs/Partitions/DATASETS/BUSI_SETUP.md)
- [Kvasir-SEG](docs/Partitions/DATASETS/KVASIR_SEG.md)
-

```bash
find /beegfs/home/mrashid/datasets \
  -maxdepth 4 -type d \
  \( -iname '*BUSI*' \
     -o -iname '*breast*' \
     -o -iname '*mvtec*' \
     -o -iname '*ultrasound*' \) \
  -print
```
