from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/scripts'))
from ground_truth import semantic_styles, draw_ground_truth
from attribution_visuals import draw_gt_boundaries


def test_same_category_instances_share_color_and_binary_mask_is_unchanged():
    labels = np.array([[0, 1, 2], [3, 0, 2]])
    gt = {'semantic_labels': True, 'mask': labels != 0, 'label_map': labels,
          'label_names': {1: 'Head', 2: 'Body', 3: 'Head'}, 'semantic_classes': ['Body', 'Head']}
    original = gt['mask'].copy()
    styles = semantic_styles(gt)
    assert styles[0]['color'] == styles[2]['color']
    assert styles[0]['color'] != styles[1]['color']
    fig, ax = plt.subplots()
    try:
        handle = draw_ground_truth(ax, gt, image=np.zeros((2, 3, 3)), legend=True)
        pixels = np.asarray(handle.get_array())
        assert pixels[0, 0, 3] == 0  # Background image stays visible.
        assert len(ax.get_legend().get_texts()) == 2
        draw_gt_boundaries(ax, gt)
        np.testing.assert_array_equal(original, gt['mask'])
    finally:
        plt.close(fig)


def test_selected_class_and_ignored_pixels_not_colored():
    labels = np.array([[1, 2], [1000, 0]])
    gt = {'semantic_labels': True, 'mask': labels == 2, 'label_map': labels,
          'label_names': {'1': 'first', '2': 'selected'}}
    assert [style['name'] for style in semantic_styles(gt)] == ['selected']
    assert semantic_styles({'mask': labels > 0, 'label_map': labels}) == []
