#!/usr/bin/env python3
"""Write tests/golden/bpt_trees_v1.npz: BPT merge trees from the v1 builder (64-bit label masks).

Run once with the v1 build of shap_bpt (before the bitset change, T1 step 0). The file stores the inputs
too, so the test does not depend on scikit-image's sample images or SLIC staying the same:
  image_<k>      uint8 RGB image, about 48x64
  partition_<k>  int64 label map (SLIC, about 12 labels, starting at 0)
  plain_left_<k>, plain_right_<k>    children of every internal node, no partition
  guided_left_<k>, guided_right_<k>  children of every internal node, with prebuilt_partitions
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def fixture_images():
    from PIL import Image
    from skimage import data
    images = [data.astronaut(), data.coffee(), data.chelsea(), data.rocket(),
              np.asarray(Image.open(ROOT / 'data/sample_images/truck.jpg').convert('RGB'))]
    return [np.asarray(Image.fromarray(image).resize((64, 48), Image.BILINEAR), dtype=np.uint8) for image in images]


def fixture_partition(image):
    from skimage.segmentation import slic
    return np.ascontiguousarray(slic(image, n_segments=12, compactness=10, start_label=0), dtype=np.int64)


def tree(image, partition=None, **options):
    import shap_bpt
    bpt = shap_bpt.build_bpt_from_image(image, **({'prebuilt_partitions': partition} if partition is not None else {}),
                                        **options)
    return np.asarray(bpt.cl_left), np.asarray(bpt.cl_right)


def main():
    arrays = {}
    for index, image in enumerate(fixture_images()):
        partition = fixture_partition(image)
        arrays[f'image_{index}'], arrays[f'partition_{index}'] = image, partition
        arrays[f'plain_left_{index}'], arrays[f'plain_right_{index}'] = tree(image)
        arrays[f'guided_left_{index}'], arrays[f'guided_right_{index}'] = tree(image, partition)
    target = Path(__file__).with_name('bpt_trees_v1.npz')
    np.savez_compressed(target, **arrays)
    print(f'Saved {target} ({len(arrays) // 6} images)')


if __name__ == '__main__':
    main()
