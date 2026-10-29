"""Attribution display conventions from fun_plot_heatmaps_filtered."""
import numpy as np


def notebook_style(heatmap):
    import shap_bpt
    values = np.asarray(heatmap)
    finite = np.abs(values[np.isfinite(values)])
    maximum = float(np.quantile(finite, .99)) if finite.size else 0.0
    # A zero quantile needs a nondegenerate color scale; do not alter the data.
    if maximum <= 0:
        maximum = 1e-12
    return {'cmap': shap_bpt.shapley_values_colormap, 'vmin': -maximum, 'vmax': maximum}


def draw_gt_boundaries(ax, mask):
    from skimage.segmentation import mark_boundaries
    mask = np.asarray(mask, dtype=bool)
    background = np.ones((*mask.shape, 4), dtype=float)
    background[..., 3] = 0
    boundary = mark_boundaries(background, mask, mode='thick', color=(0, 0, 0, 1))
    return ax.imshow(boundary, origin='upper')
