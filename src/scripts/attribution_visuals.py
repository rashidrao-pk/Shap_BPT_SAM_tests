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
    overlay = ax.imshow(boundary, origin='upper', interpolation='nearest', zorder=3)
    if mask.any():
        # Keep the outline visible when a high-resolution mask is reduced to a
        # small panel. Pixel-only boundaries can disappear during downsampling.
        from matplotlib import patheffects
        contour = ax.contour(np.arange(-1, mask.shape[1] + 1),
                             np.arange(-1, mask.shape[0] + 1),
                             np.pad(mask.astype(float), 1), levels=[.5],
                             colors='black', linewidths=1.2, zorder=5)
        effects = [patheffects.Stroke(linewidth=2.4, foreground='white'),
                   patheffects.Normal()]
        if hasattr(contour, 'set_path_effects'):
            contour.set_path_effects(effects)
        else:
            for collection in contour.collections:
                collection.set_path_effects(effects)
        ax.set_xlim(-.5, mask.shape[1] - .5)
        ax.set_ylim(mask.shape[0] - .5, -.5)
    return overlay
