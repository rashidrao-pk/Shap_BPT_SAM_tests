import copy
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'))
from budget_noninferiority import simultaneous_bounds, analyze
from budget_selection import DEFAULT_CRITERIA
from budget_ablation import parser


def test_cluster_bounds_preserve_pairing_and_are_reproducible():
    x = np.arange(12, dtype=float)/20
    block = np.column_stack([x, x, x+.2])
    first, q = simultaneous_bounds([block], resamples=2000)
    second, _ = simultaneous_bounds([block], resamples=2000)
    np.testing.assert_equal(first, second)
    assert q > 0
    assert first[0][0] == first[0][1]
    assert np.isclose(first[0][2]-first[0][0], .2)


def fixture():
    ids = [str(i) for i in range(20)]
    study = {'state': 'complete', 'runner': 'OC_ResNet', 'dataset': 'ImageNet', 'background': 'full',
        'path': '/test', 'methods': ['BPT'], 'budgets': [100, 200], 'reference_budget': 4000,
        'criteria': dict(DEFAULT_CRITERIA), 'repeats': 2, 'compute_auc': True, 'auc_points': 200,
        'splits': {'calibration': ids[:12], 'validation': ids[12:]}, 'rows': []}
    for image in ids:
        for budget in study['budgets']:
            for repeat in range(2):
                study['rows'].append({'image_id': image, 'method': 'BPT', 'budget': budget, 'repeat': repeat,
                    'informative_reference': True, 'reference_stable': True, 'relative_l1': .01,
                    'spearman': .999, 'top_jaccard': .999, 'auc_ins_difference': .001, 'auc_del_difference': .001})
    protocol = {'budgets': [100, 200], 'margins': {'relative_l1': .1},
                'bootstrap_resamples': 1000, 'alpha': .05, 'seed': 0}
    return study, protocol


def test_holdout_rejects_without_escalating_selected_budget():
    study, protocol = fixture()
    assert analyze([study], protocol)['selected_budget'] == 100
    for row in study['rows']:
        if row['image_id'] in study['splits']['validation'] and row['budget'] == 100:
            row['relative_l1'] = .5
    result = analyze([study], protocol)
    assert result['calibration_candidate'] == 100
    assert result['status'] == 'validation_failed'
    assert result['selected_budget'] is None


def test_non_full_scope_rejected_and_default_is_full():
    import pytest
    study, protocol = fixture()
    study['background'] = 'black'
    with pytest.raises(ValueError, match='full replacement'):
        analyze([study], protocol)
    for runner in ['OC_BUSI', 'OC_VGG', 'OC_PartImageNet', 'OC_CelebA']:
        assert parser(['--runner', runner]).bg_type == 'full'
