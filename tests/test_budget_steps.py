"""Consecutive-budget AUC differences, their summary, and the settled-budget rule."""
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
sys.path.insert(0, str(ROOT / 'src/scripts/explanations'))
import budget_steps


def results(values):
    """values maps image -> {budget: (AUC+, AUC-)}."""
    return pd.DataFrame([{'image_id': image, 'method': 'BPT', 'budget': budget, 'auc_ins': scores[0], 'auc_del': scores[1]}
                         for image, curve in values.items() for budget, scores in curve.items()])


def test_steps_compare_each_budget_with_the_previous_one_in_budget_order():
    frame = results({'a': {100: (.7, .2), 20: (.5, .4), 50: (.8, .3)}})
    steps = budget_steps.step_frame(frame)
    assert list(zip(steps['from_budget'], steps['to_budget'])) == [(20, 50), (50, 100)]
    assert steps['auc_ins_change'].tolist() == pytest.approx([.3, -.1])
    assert steps['auc_ins_abs_change'].tolist() == pytest.approx([.3, .1])
    assert steps['auc_del_change'].tolist() == pytest.approx([-.1, -.1])


def test_summary_averages_absolute_changes_so_opposite_signs_do_not_cancel():
    steps = budget_steps.step_frame(results({'a': {20: (.5, .5), 50: (.7, .5)}, 'b': {20: (.5, .5), 50: (.3, .5)}}))
    row = budget_steps.step_summary(steps).iloc[0]
    assert row['images'] == 2
    assert row['auc_ins_abs_change_mean'] == pytest.approx(.2)
    assert row['auc_ins_change_mean'] == pytest.approx(0)


def summary(ins, deletion=None):
    budgets = [20, 50, 100, 500]
    return pd.DataFrame({'method': 'BPT', 'from_budget': budgets[:-1], 'to_budget': budgets[1:],
                         'auc_ins_abs_change_mean': ins, 'auc_del_abs_change_mean': deletion or [0] * len(ins)})


@pytest.mark.parametrize('ins, deletion, expected', [
    ([.1, .005, .001], None, 50),
    ([.005, .005, .001], None, 20),
    ([.1, .005, .1], None, None),  # Still moving at the last step.
    ([.005, .1, .005], None, 100),  # An early dip below tolerance does not count.
    ([.005, .005, .005], [.1, .005, .005], 50),  # Both AUCs must settle.
])
def test_settled_budget(ins, deletion, expected):
    assert budget_steps.settled_budget(summary(ins, deletion), 'BPT', .01) == expected


def test_parser_defaults_match_the_yolo_sam1_study():
    args = budget_steps.parser([]).parse_args([])
    assert (args.runner, args.config, args.model, args.limit, args.bg_type) == ('OD_yolo', 'MSCOCO_epito_sam1', 'yolo11s', 5, 'full')
    assert args.budgets == [20, 50, 100, 500, 1000, 2000]


def test_rejects_a_single_budget_and_disabled_auc():
    args = budget_steps.parser([]).parse_args(['--budgets', '100'])
    with pytest.raises(ValueError, match='two budgets'):
        budget_steps.validate(args)
    args = budget_steps.parser([]).parse_args(['--no-auc'])
    with pytest.raises(ValueError, match='no-auc'):
        budget_steps.validate(args)
