"""AUC follows the explanation background unless --auc-bg-type selects another one."""
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
sys.path.insert(0, str(ROOT / 'src/scripts/explanations'))
import budget_ablation
import budget_steps
import experiment_common as common


class Brightness:
    class_names = ['bright']

    def predict_batch(self, images):
        return np.array([[np.mean(image) / 255] for image in images])


IMAGE = np.full((8, 8, 3), 128, dtype=np.uint8)


def settings(bg='full', auc=None):
    return Namespace(bg_type=bg, auc_bg_type=auc, model_batch_size=8, max_evals=1000)


@pytest.mark.parametrize('auc', [None, 'full'])
def test_default_and_matching_background_reuse_the_explanation_scoring(auc):
    def predict(_):
        return None
    assert common.auc_scoring(Brightness(), IMAGE, settings(auc=auc), 0, predict, 1., .5) == (predict, 1., .5)


@pytest.mark.parametrize('background, expected', [('black', 0), ('white', 1)])
def test_another_background_measures_its_own_f_0(background, expected):
    predict, f_s, f_0 = common.auc_scoring(Brightness(), IMAGE, settings(auc=background), 0, None, 9., 9.)
    assert f_s == pytest.approx(128 / 255)
    assert f_0 == pytest.approx(expected)
    assert predict(np.zeros((1, 8, 8), dtype=bool))[0, 0] == pytest.approx(expected)


def test_auc_changes_with_the_selected_background():
    runtime = common.load_runtime()
    heatmap = np.arange(64, dtype=float).reshape(8, 8)
    scores = {}
    for background in ['black', 'white']:
        predict, f_s, f_0 = common.auc_scoring(Brightness(), IMAGE, settings(auc=background), 0, None, 0., 0.)
        scores[background] = runtime.utx.saliency_to_auc(predict, heatmap, f_s, f_0, 0, method='del', num_samples=10)
    assert scores['black']['f_0'] != scores['white']['f_0']
    assert scores['black']['auc'] != scores['white']['auc']


@pytest.mark.parametrize('bg, auc, expected', [('full', None, 'bg_full_B1000'), ('full', 'full', 'bg_full_B1000'),
                                               ('full', 'black', 'bg_full_auc_black_B1000')])
def test_run_folder(bg, auc, expected):
    assert common.run_folder(settings(bg, auc)) == expected


def test_runs_without_the_attribute_still_work():
    assert common.run_folder(Namespace(bg_type='full', max_evals=5)) == 'bg_full_B5'


def test_default_keeps_the_signature_of_existing_runs():
    parser = common.create_parser('OC_ResNet')
    default = parser.parse_args([])
    assert common.run_signature('OC_ResNet', default) == common.run_signature('OC_ResNet', parser.parse_args(['--auc-bg-type', default.bg_type]))
    older = common.create_parser('OC_ResNet', auc_background=False).parse_args([])
    assert common.run_signature('OC_ResNet', older) == common.run_signature('OC_ResNet', default)
    assert common.run_signature('OC_ResNet', parser.parse_args(['--auc-bg-type', 'black'])) != common.run_signature('OC_ResNet', default)


def test_budget_studies_reject_the_option():
    with pytest.raises(SystemExit):
        budget_steps.parser([]).parse_args(['--auc-bg-type', 'black'])
    with pytest.raises(SystemExit):
        budget_ablation.parser(['--auc-bg-type', 'black'])


def test_saved_auc_results_can_contain_whole_curves():
    runtime = common.load_runtime()
    saved = runtime.utx.to_jsonable({'curve': [{'xs': np.array([.1, .2]), 'auc': np.float64(.5), 'one': np.array([3.])}]})
    assert saved == {'curve': [{'xs': [.1, .2], 'auc': .5, 'one': 3.}]}
