## Download SAM1 Checkpoints

- Download the [official SAM1 ViT-B checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints) once, if needed:

```bash
mkdir -p checkpoints/sam1
curl -L --fail \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -o checkpoints/sam1/sam_vit_b_01ec64.pth
```
