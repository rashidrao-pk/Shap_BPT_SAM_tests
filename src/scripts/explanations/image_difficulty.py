"""Image difficulty for choosing budget-study images, from GT and partitions only (no model).

Each feature is ranked across the candidate images; the difficulty score is the mean of those
percentile ranks, oriented so that higher means harder. The pool is split into thirds (easy,
medium, hard). Which features really predict slow convergence is an empirical question:
check_features() correlates them with how far each image's AUC still is from its final value.
"""
from pathlib import Path

import numpy as np

# Feature -> +1 if a larger value should be harder, -1 if a smaller value should be harder.
FEATURES = {
    'regions': +1,           # SAM_c cells: more coalitions to sample
    'objects': +1,           # GT objects: clutter
    'categories': +1,        # Distinct GT classes
    'max_same_class': +1,    # Instances of the most repeated class: evidence spread over regions
    'smallest_object': -1,   # Smallest GT object, share of the image: needs fine splits
    'mean_object': -1,       # Mean GT object size, share of the image
    'partition_fit': -1,     # Mean best IoU of the partition with each GT object
}
STRATA = ['easy', 'medium', 'hard']


def image_features(config, record):
    """Features of one image. Images without GT objects get counts of 0 and no size or fit."""
    from evaluate_partitions import gt_objects, load_partition, maximize_iou
    partition = load_partition(Path(record['image_path']), Path(record['partition_path']))
    shape, pixels = partition.shape, partition.size
    sizes, fits, names = [], [], []
    for name, mask, valid in gt_objects(config, Path(record['image_path']), shape):
        if not mask.any():
            continue
        selected = np.ones(shape, dtype=bool) if valid is None else valid
        names.append(str(name))
        sizes.append(mask.sum() / pixels)
        fits.append(maximize_iou(mask[selected], partition[selected])[0])
    counts = {name: names.count(name) for name in set(names)}
    return {'image_id': record['image_id'], 'regions': int(np.count_nonzero(np.unique(partition))),
            'objects': len(names), 'categories': len(counts), 'max_same_class': max(counts.values(), default=0),
            'smallest_object': float(min(sizes)) if sizes else np.nan,
            'mean_object': float(np.mean(sizes)) if sizes else np.nan,
            'partition_fit': float(np.mean(fits)) if fits else np.nan}


def feature_table(config, records, cache_path=None, verbose=True):
    """Features for every record, reusing the ones already saved in cache_path."""
    import pandas as pd
    saved = pd.DataFrame()
    if cache_path is not None and Path(cache_path).is_file():
        saved = pd.read_csv(cache_path, dtype={'image_id': str})
    known = set(saved['image_id']) if len(saved) else set()
    missing = [record for record in records if record['image_id'] not in known]
    rows = []
    for number, record in enumerate(missing, 1):
        try:
            rows.append(image_features(config, record))
        except (OSError, ValueError, KeyError) as error:  # One unreadable image must not stop the selection.
            print(f"{record['image_id']}: difficulty skipped ({type(error).__name__}: {error})", flush=True)
        if verbose and (number % 100 == 0 or number == len(missing)):
            print(f'Difficulty features: {number}/{len(missing)} new images', flush=True)
    table = pd.concat([saved, pd.DataFrame(rows)], ignore_index=True) if rows else saved
    if cache_path is not None and rows:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_path, index=False)
    wanted = {record['image_id'] for record in records}
    return table[table['image_id'].isin(wanted)].reset_index(drop=True)


def score(table):
    """Mean oriented percentile rank (0 easy .. 1 hard) and a stratum per image, within this pool."""
    table = table.copy()
    # Percentile rank per feature, flipped where a smaller value is harder; missing values are skipped.
    ranks = [table[name].rank(pct=True, ascending=direction > 0).to_numpy(dtype=float)
             for name, direction in FEATURES.items() if name in table]
    table['difficulty'] = np.nanmean(np.vstack(ranks), axis=0)
    order = table['difficulty'].rank(method='first')
    table['stratum'] = [STRATA[min(int(3 * (value - 1) / len(table)), 2)] for value in order]
    return table


def select(table, limit, seed, mode):
    """Image IDs for the study. A larger limit keeps the earlier images (prefix-stable).

    stratified: alternate hard, medium, easy, each in its own seeded random order.
    hard: the hardest third, in seeded random order.
    """
    rng = np.random.default_rng(seed)
    groups = {stratum: list(table.loc[table['stratum'] == stratum, 'image_id']) for stratum in STRATA}
    for stratum in STRATA:
        groups[stratum] = [groups[stratum][index] for index in rng.permutation(len(groups[stratum]))]
    if mode == 'hard':
        order = groups['hard']
    elif mode == 'stratified':
        order = [group[index] for index in range(max(map(len, groups.values()), default=0))
                 for group in (groups['hard'], groups['medium'], groups['easy']) if index < len(group)]
    else:
        raise ValueError(f'Unknown selection {mode}')
    if limit > len(order):
        raise ValueError(f'--limit {limit} is more than the {len(order)} {mode} images available')
    return order[:limit]


def check_features(results, features, budget, metrics=('auc_ins', 'auc_del')):
    """Spearman correlation of each feature with |AUC(budget) - AUC(largest budget)| per image.

    A positive correlation for a +1 feature (or negative for a -1 feature) supports using it.
    """
    import pandas as pd
    largest = results['budget'].max()
    if budget not in set(results['budget']) or budget == largest:
        raise ValueError(f'--check-difficulty needs a tested budget below the largest ({largest})')
    rows = []
    for method, group in results.groupby('method', sort=False):
        pivot = group.pivot_table(index='image_id', columns='budget', values=list(metrics))
        for metric in metrics:
            gap = (pivot[(metric, budget)] - pivot[(metric, largest)]).abs().rename('gap')
            merged = features.set_index('image_id').join(gap, how='inner')
            for name, direction in FEATURES.items():
                pair = merged[[name, 'gap']].dropna()
                varies = len(pair) > 2 and pair[name].nunique() > 1 and pair['gap'].nunique() > 1
                rho = pair[name].corr(pair['gap'], method='spearman') if varies else np.nan
                rows.append({'method': method, 'metric': metric, 'feature': name, 'images': len(pair),
                             'spearman': rho, 'expected_sign': direction,
                             'agrees': bool(np.sign(rho) == direction) if np.isfinite(rho) else None})
    return pd.DataFrame(rows)
