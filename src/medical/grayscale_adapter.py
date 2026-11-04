"""Preserve ultrasound intensities when providing three-channel model inputs."""
import numpy as np
from PIL import Image


def to_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError('Expected uint8 BUSI intensities; no implicit per-image normalization')
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert('RGB')
