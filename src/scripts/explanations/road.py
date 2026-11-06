"""ROAD: remove pixels by attribution rank, infill them from neighbours, and re-score the model.

Rong et al., "A Consistent and Efficient Evaluation Strategy for Attribution Methods", ICML 2022.
The imputer follows the equations of the authors' NoisyLinearImputer (MIT licence), with the
sparse matrix assembled in one step. Every method removes the same number of pixels at each
fraction, so regions with equal attribution do not change the evaluation points.
"""
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import splu

FRACTIONS = (.1, .2, .3, .4, .5, .6, .7, .8, .9)
NOISE = .01  # Absolute, on 0-1 images, as in the authors' implementation.
SHORT_SIDE = 256
ORDERS = ('morf', 'lerf')  # Most / least relevant pixels first.
# Each missing pixel is the weighted mean of its eight neighbours.
NEIGHBOURS = (((1, 1), 1/12), ((0, 1), 1/6), ((-1, 1), 1/12), ((1, -1), 1/12),
              ((0, -1), 1/6), ((-1, -1), 1/12), ((1, 0), 1/6), ((-1, 0), 1/6))


def working_size(height, width, short_side):
    """Scale down only, keeping the aspect ratio; 0 or None keeps the original resolution."""
    if not short_side or min(height, width) <= short_side:
        return height, width
    scale = short_side / min(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def resize_inputs(image, heatmap, short_side):
    height, width = working_size(*image.shape[:2], short_side)
    if (height, width) == image.shape[:2]:
        return image, np.asarray(heatmap, dtype=float)
    pixels = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))
    # BOX averages attribution within each new pixel, so no region is dropped.
    values = Image.fromarray(np.asarray(heatmap, dtype=np.float32)).resize((width, height), Image.Resampling.BOX)
    return pixels, np.asarray(values, dtype=float)


def impute(image, keep, noise=NOISE, rng=None):
    """Fill pixels where keep is False. image is [H, W, C] in 0-1; kept pixels are unchanged."""
    keep = np.asarray(keep, dtype=bool)
    if keep.all():
        return image.copy()
    if not keep.any():
        raise ValueError('ROAD needs at least one kept pixel')
    height, width = keep.shape
    missing_y, missing_x = np.nonzero(~keep)
    count = len(missing_y)
    variable = np.full(keep.shape, -1, dtype=np.int64)
    variable[missing_y, missing_x] = np.arange(count)
    right = np.zeros((count, image.shape[2]))
    diagonal = np.zeros(count)
    rows, columns, values = [], [], []
    for (dy, dx), weight in NEIGHBOURS:
        y, x = missing_y + dy, missing_x + dx
        inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
        ids, y, x = np.nonzero(inside)[0], y[inside], x[inside]
        diagonal[ids] += weight  # Image edges have fewer neighbours, with less total weight.
        known = keep[y, x]
        right[ids[known]] += weight * image[y[known], x[known]]
        rows.append(ids[~known])
        columns.append(variable[y[~known], x[~known]])
        values.append(np.full((~known).sum(), -weight))
    rows.append(np.arange(count))
    columns.append(np.arange(count))
    values.append(diagonal)
    system = sparse.csc_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                               shape=(count, count))
    solution = splu(system).solve(right)
    if noise:
        solution = solution + noise * (rng or np.random.default_rng()).standard_normal(solution.shape)
    result = image.copy()
    result[missing_y, missing_x] = solution
    return result


def removal_order(heatmap, tie_break):
    """Pixels from most to least relevant, with one shared random order for equal values."""
    return np.lexsort((tie_break.ravel(), -np.asarray(heatmap, dtype=float).ravel()))


def keep_mask(order, fraction, shape, direction):
    """Exactly round(fraction * pixels) pixels are removed, whatever the attribution's structure."""
    removed = int(round(fraction * order.size))
    selected = order[:removed] if direction == 'morf' else order[order.size - removed:]
    keep = np.ones(order.size, dtype=bool)
    keep[selected] = False
    return keep.reshape(shape)


def area(fractions, scores, baseline):
    """Mean score across removal levels, from the unmodified image to the last fraction."""
    xs = np.array([0, *fractions], dtype=float)
    ys = np.array([baseline, *scores], dtype=float)
    return float(((ys[1:] + ys[:-1]) / 2 * np.diff(xs)).sum() / (xs[-1] - xs[0]))


def evaluate(predict, image, heatmap, target, fractions=FRACTIONS, noise=NOISE, seed=0, short_side=SHORT_SIDE):
    """Score one attribution map in both directions.

    predict takes uint8 images and returns [N, classes]. Noise and tie-breaking depend only on
    seed, so methods compared on one image share them.
    """
    if not len(fractions) or min(fractions) <= 0 or max(fractions) >= 1:
        raise ValueError('ROAD fractions must lie strictly between 0 and 1')
    fractions = sorted(fractions)
    pixels, values = resize_inputs(image, heatmap, short_side)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite attribution')
    baseline = float(np.asarray(predict([pixels]))[0, target])
    scaled = pixels.astype(float) / 255
    order = removal_order(values, np.random.default_rng(seed).random(values.shape))
    result = {'fractions': list(fractions), 'baseline': baseline, 'size': list(values.shape)}
    for direction in ORDERS:
        rng = np.random.default_rng(seed + 1)
        images = []
        for fraction in fractions:
            filled = impute(scaled, keep_mask(order, fraction, values.shape, direction), noise, rng)
            images.append(np.clip(np.rint(filled * 255), 0, 255).astype(np.uint8))
        scores = np.asarray(predict(images), dtype=float)[:, target]
        result[f'{direction}_scores'] = scores.tolist()
        result[direction] = area(fractions, scores, baseline)
    result['gap'] = result['lerf'] - result['morf']
    # Relative values compare images with different confidence; logit scores can be negative.
    for name in [*ORDERS, 'gap']:
        result[f'{name}_rel'] = result[name] / baseline if baseline > 0 else None
    return result
