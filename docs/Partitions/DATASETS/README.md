### DATASETS

- [BUSI](BUSI_SETUP.md)
- [Kvasir-SEG](../../DATASETS/KVASIR_SEG.md)
- [All datasets](../../DATASETS/README.md)

```bash
find /beegfs/home/mrashid/datasets \
  -maxdepth 4 -type d \
  \( -iname '*BUSI*' \
     -o -iname '*breast*' \
     -o -iname '*mvtec*' \
     -o -iname '*ultrasound*' \) \
  -print
```
