"""Colours for partition previews: label 0 (pixels no SAM mask covers) is black, every other label a distinct colour."""
import numpy as np

BACKGROUND = (0.0, 0.0, 0.0)


def label_palette(count):
    """`count` RGB colours from turbo, never near black, ordered so that consecutive labels differ strongly."""
    if count <= 0:
        return np.zeros((0, 3))
    import matplotlib.pyplot as plt
    positions = (0.5 + np.arange(count) * 0.6180339887) % 1.0  # Golden-ratio steps spread neighbours apart.
    return plt.colormaps['turbo'](0.12 + 0.83 * positions)[:, :3]


def colorize(labels):
    """HxW integer labels -> HxWx3 floats in [0, 1]; label 0 is black."""
    labels = np.asarray(labels)
    unique, inverse = np.unique(labels, return_inverse=True)
    foreground = unique != 0
    colors = np.zeros((len(unique), 3))
    colors[foreground] = label_palette(int(foreground.sum()))
    colors[~foreground] = BACKGROUND
    return colors[inverse.reshape(labels.shape)]
