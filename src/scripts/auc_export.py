"""Flatten scalar AUC variants for tables; full curves stay in metrics JSON."""

AUC_VARIANTS = ('auc', 'auc_r', 'auc_mae', 'auc_mse', 'auc_rmse',
                'auc_clip', 'auc_clipr', 'auc_adj', 'auc_adjr')
AUC_METADATA = ('requested_points', 'evaluated_points', 'f_0', 'f_S', 'flipped',
                'score_delta_denominator', 'predicted_cls')


def auc_columns(result):
    columns = {}
    for direction in ('ins', 'del'):
        curve = result.get('auc_' + direction)
        if not isinstance(curve, dict):
            continue
        prefix = 'auc_' + direction
        if 'auc_clipr' in curve:
            columns[prefix] = float(curve['auc_clipr'])  # Historical plotting/summary alias.
        for key in AUC_VARIANTS:
            if key in curve:
                suffix = '_raw' if key == 'auc' else key[3:]
                columns[prefix + suffix] = float(curve[key])
        for key in AUC_METADATA:
            if key in curve:
                value = curve[key]
                columns[prefix + '_' + key] = value.item() if hasattr(value, 'item') else value
    return columns


def enrich_auc_rows(rows, results):
    """Enrich resumed/exported rows without model inference or changing other metrics."""
    by_method = {item.get('label', item.get('exp_code', item.get('method'))): item
                 for item in results}
    for row in rows:
        result = by_method.get(row.get('method'))
        if result is not None:
            row.update(auc_columns(result))
    return rows
