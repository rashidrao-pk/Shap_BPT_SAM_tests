"""Native h-Shap and EAC baselines (TO_DO T3a, T3b)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'))
import hshap_native as hshap  # noqa: E402
import eac_native as eac  # noqa: E402
import all_methods  # noqa: E402


class Counter:
    """Masked predictor whose class-0 score is the visible share of a hot block, plus a little elsewhere."""

    def __init__(self, shape=(64, 64), hot=(40, 48, 8, 16)):
        self.weight = np.full(shape, .001)
        y0, y1, x0, x1 = hot
        self.weight[y0:y1, x0:x1] = 1 / ((y1 - y0) * (x1 - x0))
        self.calls = 0

    def __call__(self, masks):
        self.calls += len(masks)
        score = (np.asarray(masks) * self.weight).sum(axis=(1, 2))
        return np.stack([score, -score], axis=1)


def test_exact_shapley_of_four_players():
    weights = np.array([.5, .1, -.2, .3])
    values = [sum(weights[i] for i in range(4) if coalition >> i & 1) for coalition in range(16)]
    np.testing.assert_allclose(hshap.shapley_4(values), weights)
    interaction = [1.0 if coalition & 3 == 3 else 0.0 for coalition in range(16)]  # players 0 AND 1
    np.testing.assert_allclose(hshap.shapley_4(interaction), [.5, .5, 0, 0])


def test_hshap_finds_the_hot_block_within_budget():
    predict = Counter()
    heatmap, info = hshap.explain_budgeted(predict, 0, (64, 64), budget=200, min_size=8)
    assert info['realised_evals'] <= 200 and predict.calls == info['realised_evals']
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    assert 40 <= y < 48 and 8 <= x < 16
    assert heatmap[:32, 32:].max() == 0  # The far quadrant is pruned
    small = hshap.explain_budgeted(Counter(), 0, (64, 64), budget=16, min_size=8)[1]
    assert small['realised_evals'] <= 16 and small['nodes'] <= info['nodes']


def test_hshap_cycle_spinning_splits_the_budget():
    predict = Counter()
    heatmap, info = hshap.explain_cycle_spinning(predict, 0, (64, 64), budget=500, min_size=8)
    assert heatmap.shape == (64, 64) and info['realised_evals'] <= 500 and len(info['shifts']) == 5
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    assert 36 <= y < 52 and 4 <= x < 20


@pytest.mark.parametrize('estimator', ['eac', 'shapley'])
def test_eac_recovers_additive_region_weights(estimator):
    pytest.importorskip('torch')
    labels = np.repeat(np.repeat(np.arange(4).reshape(2, 2), 8, axis=0), 8, axis=1)  # 4 regions of 8x8
    weights = np.array([.1, .7, .3, .5])
    calls = []

    def predict(masks):
        calls.append(len(masks))
        share = np.stack([(np.asarray(masks) & (labels == r)).mean(axis=(1, 2)) * 4 for r in range(4)], axis=1)
        score = share @ weights
        return np.stack([score, 1 - score], axis=1)

    heatmap, info = eac.explain(predict, labels, 0, budget=64, mc_samples=500, estimator=estimator, batch_size=16)
    assert sum(calls) == 64 == info['realised_evals']
    assert np.argsort(info['concept_values']).tolist() == np.argsort(weights).tolist()
    assert info['surrogate_r2'] > .9
    assert heatmap[0, 0] == info['concept_values'][0]  # Region value on every pixel (no area division)


def test_extra_methods_are_named_and_keep_old_signatures():
    parser = argparse.ArgumentParser()
    all_methods.add_arguments(parser)
    parser.add_argument('--max-evals', type=int, default=1000)
    import guide_options
    guide_options.add_arguments(parser)
    plain = parser.parse_args([])
    extra = parser.parse_args(['--extra-methods', 'hshap', 'hshap-cs', 'eac'])
    names = all_methods.method_names(extra)
    assert names[3:6] == ['hShap-1000', 'hShap-CS-1000', 'EAC-1000']  # Right after the three partition methods
    assert 'EAC-1000' not in all_methods.method_names(plain)
    legacy = argparse.Namespace(**{k: v for k, v in vars(plain).items() if k not in all_methods.EXTRA_OPTIONS})
    assert all_methods.comparison_signature(plain) == all_methods.comparison_signature(legacy)
    assert all_methods.comparison_signature(extra) != all_methods.comparison_signature(plain)


def test_compute_extra_dispatches_the_new_methods(monkeypatch):
    pytest.importorskip('torch')
    predict = Counter((32, 32), hot=(16, 24, 0, 8))
    labels = np.zeros((32, 32), dtype=int)
    labels[16:, :16] = 1
    args = argparse.Namespace(eval_batch_size=16, explanation_seed=0, hshap_min_size=4, eac_mc_samples=200,
                              eac_estimator='eac', verbose_level='low')
    monkeypatch.setattr(all_methods, 'method_names', lambda _: ['hShap-64', 'hShap-CS-100', 'EAC-32'])
    results = {name: (values, info) for name, values, info, _ in all_methods.compute_extra(
        None, args, None, np.zeros((32, 32, 3)), None, None, None, 0, labels=labels, predict=predict)}
    for name, (values, info) in results.items():
        assert info['status'] == 'complete' and values.shape == (1, 32, 32), name
        assert info['realised_evals'] <= int(name.rsplit('-', 1)[1])
    assert results['EAC-32'][1]['concept_values'][1] > results['EAC-32'][1]['concept_values'][0]
