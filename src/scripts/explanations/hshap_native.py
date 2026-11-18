"""h-Shap (Teneggi, Luster and Sulam, "Fast hierarchical games for image explanations", IEEE TPAMI 2022),
re-implemented on this repository's masked predictor, so it shares the value function of the other methods and
also runs on detectors and vision-language models (TO_DO T3a).

The image is a quadtree. At each node, the four quadrants are the players of a game in which everything
outside the node is replaced by the background: the 2^4 = 16 coalitions are evaluated (each once, cached
across the whole explanation) and the exact Shapley value of every quadrant is computed. Quadrants with a value
above `tol` are explored further until a quadrant can no longer be split into children of at least `min_size`
pixels per side; such a quadrant is a leaf and keeps its value. Pruned and unexplored pixels get 0.

Per-pixel convention: a leaf's Shapley value is given to every one of its pixels, as
`utils_xai.kernel_shap_partition_regions` does for SAM regions (no division by area).

Budget: the cost depends on `tol`, not on a sample count. `explain_budgeted` binary-searches `tol`, keeping
`min_size` fixed, for the most detailed map whose number of distinct coalitions stays within the budget. Every
coalition evaluated during the search counts against the budget (the cache is shared), so the total cost never
exceeds it.
"""
from math import factorial

import numpy as np

# Shapley weight of a coalition of size s among the other 3 players: s! (4 - s - 1)! / 4!
_WEIGHTS = [factorial(s) * factorial(3 - s) / factorial(4) for s in range(4)]


class BudgetExceeded(Exception):
    pass


def quadrants(node):
    y0, y1, x0, x1 = node
    ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
    return [(y0, ym, x0, xm), (y0, ym, xm, x1), (ym, y1, x0, xm), (ym, y1, xm, x1)]


def shapley_4(values):
    """Exact Shapley values of 4 players from the 16 coalition values, indexed by bitmask."""
    phi = np.zeros(4)
    for player in range(4):
        bit = 1 << player
        for coalition in range(16):
            if not coalition & bit:
                phi[player] += _WEIGHTS[bin(coalition).count('1')] * (values[coalition | bit] - values[coalition])
    return phi


class HShap:
    """One quadtree explanation; `shift` moves the grid cyclically (cycle spinning)."""

    def __init__(self, predict, target, shape, max_evals=None, shift=(0, 0)):
        self.predict, self.target, self.shape = predict, target, tuple(shape)
        self.max_evals, self.shift = max_evals, tuple(shift)
        self.cache = {}

    def _mask(self, rectangles):
        mask = np.zeros(self.shape, dtype=bool)
        for y0, y1, x0, x1 in rectangles:
            mask[y0:y1, x0:x1] = True
        return np.roll(mask, self.shift, axis=(0, 1)) if any(self.shift) else mask

    def _values(self, children):
        keys = [frozenset(children[i] for i in range(4) if coalition >> i & 1) for coalition in range(16)]
        new = [key for key in dict.fromkeys(keys) if key not in self.cache]
        if self.max_evals is not None and len(self.cache) + len(new) > self.max_evals:
            raise BudgetExceeded
        if new:
            scores = self.predict(np.stack([self._mask(key) for key in new]))[:, self.target]
            self.cache.update(zip(new, np.asarray(scores, dtype=float)))
        return [self.cache[key] for key in keys]

    def explain(self, tol, min_size):
        height, width = self.shape
        heatmap = np.zeros(self.shape, dtype=float)
        stack, nodes, leaves = [(0, height, 0, width)], 0, 0
        while stack:
            node = stack.pop()
            children = quadrants(node)
            phi = shapley_4(self._values(children))
            nodes += 1
            for child, value in zip(children, phi):
                if value <= tol:
                    continue  # Pruned: 0
                y0, y1, x0, x1 = child
                if min(y1 - y0, x1 - x0) // 2 < min_size:
                    heatmap[y0:y1, x0:x1] = value
                    leaves += 1
                else:
                    stack.append(child)
        if any(self.shift):
            heatmap = np.roll(heatmap, self.shift, axis=(0, 1))
        return heatmap, {'nodes': nodes, 'leaves': leaves}


def explain_budgeted(predict, target, shape, budget, min_size=8, steps=12, shift=(0, 0)):
    """Most detailed h-Shap map within `budget` coalitions; returns (H x W map, info)."""
    if budget < 16:
        raise ValueError('h-Shap needs a budget of at least 16 coalitions (one quadtree node)')
    game = HShap(predict, target, shape, max_evals=budget, shift=shift)
    root_phi = shapley_4(game._values(quadrants((0, shape[0], 0, shape[1]))))
    high = max(float(root_phi.max()), 0.0)
    best = (high, *game.explain(high, min_size))  # Root only: always fits
    low = 0.0
    for _ in range(steps):
        tol = (low + high) / 2 if high > 0 else 0.0
        try:
            best = (tol, *game.explain(tol, min_size))
            high = tol
        except BudgetExceeded:
            low = tol
        if high - low <= 1e-12:
            break
    tol, heatmap, details = best
    return heatmap, {**details, 'tol': tol, 'min_size': min_size, 'realised_evals': len(game.cache),
                     'budget': budget, 'per_pixel': 'leaf Shapley value on every leaf pixel (no area division)'}


def explain_cycle_spinning(predict, target, shape, budget, min_size=8, n_shifts=4):
    """hShap-CS: the average of the unshifted map and maps on grids shifted by a quarter of the shorter side
    (right, down, left, up). The budget is split equally between the grids."""
    radius = min(shape) // 4
    shifts = [(0, 0), (0, radius), (radius, 0), (0, -radius), (-radius, 0)][:n_shifts + 1]
    share = budget // len(shifts)
    maps, total, tols = [], 0, []
    for shift in shifts:
        heatmap, details = explain_budgeted(predict, target, shape, share, min_size, shift=shift)
        maps.append(heatmap)
        total += details['realised_evals']
        tols.append(details['tol'])
    return np.mean(maps, axis=0), {'realised_evals': total, 'budget': budget, 'min_size': min_size,
                                    'shifts': [list(shift) for shift in shifts], 'tol': tols,
                                    'per_pixel': 'mean over shifted grids of leaf values (no area division)'}
