"""Localization diagnostics; insertion/deletion use the shared XAI evaluator."""
import numpy as np


def pointing_accuracy(attribution, mask):
    values, mask = np.asarray(attribution), np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError('Attribution and lesion mask shapes differ')
    if not mask.any():
        return None  # A normal image has no lesion to point to.
    if not np.isfinite(values).all():
        raise ValueError('Non-finite attribution')
    peaks = values == values.max()
    return float(mask[peaks].mean())  # Average ties rather than choosing an arbitrary pixel.
