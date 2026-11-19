### DATASETS

- [BUSI](BUSI_SETUP.md)
- [MS-COCO](MSCOCO.md)
- [ImageNet](ImageNet.md)
- [CAMO](CAMO.md)
- [FunnyBirds](FUNNYBIRDS.md)
- [PACO](PACO.md)
- [Kvasir-SEG](KVASIR_SEG.md)
- [ISIC 2018](ISIC2018.md)
- [DOTA](DOTA.md)
- [RefCOCO](REFCOCO.md)
- [All datasets](../../SETUP/DATASETS/README.md)

```bash
find /beegfs/home/mrashid/datasets \
  -maxdepth 4 -type d \
  \( -iname '*BUSI*' \
     -o -iname '*breast*' \
     -o -iname '*mvtec*' \
     -o -iname '*ultrasound*' \) \
  -print
```
