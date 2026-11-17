import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'),
                str(Path(__file__).resolve().parents[1] / 'src/scripts')]
import image_difficulty as difficulty  # noqa: E402


def table(n=9):
    """Image i gets harder with i on every feature."""
    i = np.arange(n)
    return pd.DataFrame({'image_id': [f'im{k}' for k in i], 'regions': 10 + i, 'objects': i, 'categories': i,
                         'max_same_class': i, 'smallest_object': 1 / (1 + i), 'mean_object': 1 / (1 + i),
                         'partition_fit': 1 - i / n})


def test_harder_features_give_a_higher_score_and_equal_thirds():
    scored = difficulty.score(table())
    assert scored['difficulty'].is_monotonic_increasing
    assert list(scored['stratum']) == ['easy'] * 3 + ['medium'] * 3 + ['hard'] * 3


def test_missing_sizes_do_not_break_the_score():
    data = table()
    data.loc[0, ['smallest_object', 'mean_object', 'partition_fit']] = np.nan  # Image without GT objects.
    assert np.isfinite(difficulty.score(data)['difficulty']).all()


def test_stratified_alternates_strata_and_a_larger_limit_keeps_earlier_images():
    scored = difficulty.score(table(30))
    strata = dict(zip(scored['image_id'], scored['stratum']))
    six = difficulty.select(scored, 6, 0, 'stratified')
    assert [strata[image] for image in six] == ['hard', 'medium', 'easy'] * 2
    assert difficulty.select(scored, 9, 0, 'stratified')[:6] == six


def test_hard_takes_only_the_hardest_third_and_refuses_more():
    scored = difficulty.score(table(30))
    chosen = difficulty.select(scored, 10, 0, 'hard')
    assert set(chosen) == set(scored.loc[scored['stratum'] == 'hard', 'image_id'])
    with pytest.raises(ValueError):
        difficulty.select(scored, 11, 0, 'hard')


def test_feature_table_reuses_saved_features(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(difficulty, 'image_features',
                        lambda config, record: calls.append(record['image_id']) or {'image_id': record['image_id'], 'regions': 5})
    records = [{'image_id': '001'}, {'image_id': '002'}]
    cache = tmp_path / 'image_difficulty.csv'
    difficulty.feature_table({}, records, cache, verbose=False)
    result = difficulty.feature_table({}, [*records, {'image_id': '003'}], cache, verbose=False)
    assert calls == ['001', '002', '003']
    assert list(result['image_id']) == ['001', '002', '003']  # Leading zeros kept.


def test_check_finds_a_feature_that_tracks_the_remaining_auc_gap():
    rows = []
    for k in range(8):
        for budget, auc in [(1000, .5 + .01 * k), (8000, .5)]:
            rows.append({'image_id': f'im{k}', 'method': 'BPT', 'budget': budget, 'auc_ins': auc, 'auc_del': .1})
    check = difficulty.check_features(pd.DataFrame(rows), table(8), 1000)
    regions = check[(check['feature'] == 'regions') & (check['metric'] == 'auc_ins')].iloc[0]
    assert regions['spearman'] == pytest.approx(1) and regions['agrees']
    with pytest.raises(ValueError):
        difficulty.check_features(pd.DataFrame(rows), table(8), 8000)
