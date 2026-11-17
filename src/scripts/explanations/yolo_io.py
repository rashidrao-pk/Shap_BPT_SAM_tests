"""Channel order for Ultralytics YOLO inputs.

Ultralytics treats NumPy image arrays as BGR (OpenCV order), while this pipeline holds images as RGB.
Every NumPy image must therefore be reversed along its channel axis before model.predict; file paths and
PIL images are read by Ultralytics itself and need no change.

YOLO_INPUT_VERSION is part of every cache signature for YOLO results:
    1: RGB arrays passed as they were, so red and blue were swapped (results before 2026-11-17)
    2: RGB arrays converted to BGR here
"""
import numpy as np

YOLO_INPUT_VERSION = 2


def to_bgr(image):
    """One image for Ultralytics: RGB NumPy arrays become contiguous BGR; anything else is returned as is."""
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[-1] == 3:
        return np.ascontiguousarray(image[..., ::-1])
    return image


def ultralytics_input(images):
    """A single image or a list/array of images, converted with to_bgr."""
    if isinstance(images, np.ndarray) and images.ndim == 4:
        return [to_bgr(image) for image in images]
    if isinstance(images, (list, tuple)):
        return [to_bgr(image) for image in images]
    return to_bgr(images)
