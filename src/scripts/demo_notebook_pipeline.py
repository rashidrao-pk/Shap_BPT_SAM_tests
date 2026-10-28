#!/usr/bin/env python3
"""Test notebook postprocessing on three bundled photos with non-SAM mask fixtures."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data

from run_notebook_pipeline import process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/notebook_pipeline_demo'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, image in [('astronaut', data.astronaut()), ('coffee', data.coffee()), ('chelsea', data.chelsea())]:
        image = np.asarray(Image.fromarray(image).resize((128, 96)))
        path = args.output_dir / f'{name}.png'
        Image.fromarray(image).save(path)
        # Intentionally simple, overlapping fixtures. These are NOT SAM predictions.
        brightness = image.mean(axis=2)
        masks = np.stack([brightness > 55, brightness > 125, brightness > 190,
                          image[:, :, 0] > image[:, :, 1] + 15.0,
                          image[:, :, 2] > 110])
        scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
        result = process(path, masks, scores, args.output_dir / name, select_count=4)
        result['mask_source'] = 'Image-derived threshold fixtures; not SAM predictions'
        (args.output_dir / name / f'{name}_notebook_results.json').write_text(json.dumps(result, indent=2) + '\n')
        summary.append({'image': name, 'mask_source': result['mask_source'],
                        'selected_masks': result['n_coverage_masks'],
                        'statistics': result['statistics']})
    (args.output_dir / 'demo_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Three-photo postprocessing demo saved to {args.output_dir}; no SAM inference performed.')


if __name__ == '__main__':
    main()
