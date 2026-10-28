#!/usr/bin/env python3
"""Save a headless comparison of partition maps and their image overlays."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


def render(image_path, partition_paths, output, alpha=0.45):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    paths = [Path(path) for path in partition_paths]
    if not paths:
        raise ValueError('No partition maps selected')
    fig, axes = plt.subplots(2, len(paths) + 1, squeeze=False,
                             figsize=(4 * (len(paths) + 1), 7))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original')
    statistics = []
    try:
        for column, path in enumerate(paths, 1):
            labels = np.load(path, allow_pickle=False)
            if labels.ndim != 2 or labels.shape != image.shape[:2] or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f'{path}: expected integer labels of shape {image.shape[:2]}')
            unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
            colors = plt.colormaps['turbo'](np.linspace(0.05, 0.95, len(unique)))[:, :3]
            colored = colors[inverse.reshape(labels.shape)]
            axes[0, column].imshow(colored)
            axes[0, column].set_title(f'{path.stem}\n{len(unique)} labels', fontsize=9)
            overlay = (1 - alpha) * (image / 255.) + alpha * colored
            axes[1, column].imshow(mark_boundaries(overlay, labels, color=(1, 1, 1)))
            axes[1, column].set_title('Overlay + boundaries')
            statistics.append({'path': str(path), 'shape': list(labels.shape),
                               'label_count': len(unique),
                               'zero_pixels': int(np.count_nonzero(labels == 0)),
                               'min_region_pixels': int(counts.min()),
                               'max_region_pixels': int(counts.max())})
        for ax in axes.ravel():
            ax.axis('off')
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)
        output.with_suffix('.json').write_text(json.dumps(statistics, indent=2) + '\n')
    finally:
        plt.close(fig)
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--partitions', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=0.45)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1')
    print(json.dumps(render(args.image, args.partitions, args.output, args.alpha), indent=2))


if __name__ == '__main__':
    main()
