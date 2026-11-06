import copy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/scripts/explanations'))
from budget_interactive import build_args, statistical_report
from budget_reuse import validate_reuse
import budget_ablation as pilot
from test_budget_noninferiority import fixture


def test_builder_keeps_original_plan_and_excludes_kernelshap(tmp_path):
    folder = tmp_path/'MSCOCO_sam1_vit_b_62_compact-sam_200/bg_full/full_budget'
    folder.mkdir(parents=True)
    source = {'runner': 'OD_yolo', 'background': 'full', 'methods': list(pilot.METHODS),
              'settings': {'limit': 20, 'image_ids': [str(i) for i in range(20)], 'bg_type': 'full'},
              'splits': {'calibration': list(map(str, range(12))), 'validation': list(map(str, range(12, 20)))}}
    (folder/'study.json').write_text(json.dumps(source))
    args = build_args(folder, images=3)
    assert args.methods == ['BPT', 'BPT+SAM_c']
    assert args.limit == 20 and args.image_ids == source['settings']['image_ids']
    assert args.stop_after_images == 3 and args.statistical_test
    assert build_args(folder, final=True).stop_after_images is None
    with pytest.raises(ValueError, match='--final'):
        build_args(folder, images=13)


def test_reuse_rejects_changed_scoring_or_inputs(tmp_path):
    source, _ = fixture()
    source['settings'] = {'model': 'resnet50', 'threshold': 0}
    source['signature'] = 'original'
    (tmp_path/'study.json').write_text(json.dumps(source))
    current = copy.deepcopy(source)
    assert validate_reuse(current, tmp_path)['signature'] == 'original'
    current['settings']['threshold'] = .5
    with pytest.raises(ValueError, match='settings.threshold'):
        validate_reuse(current, tmp_path)
    current = copy.deepcopy(source); current['inputs'] = ['changed']
    with pytest.raises(ValueError, match='inputs'):
        validate_reuse(current, tmp_path)


def test_statistical_report_is_limited_to_actual_methods(tmp_path):
    study, protocol = fixture()
    study['statistical_protocol'] = protocol
    result = statistical_report(tmp_path, study)
    assert result['selected_budget'] == 100
    assert (tmp_path/'statistical_bounds.csv').is_file()
    assert 'not attribution correctness' in (tmp_path/'statistical_conclusion.md').read_text()
    assert pilot.parser([]).methods == ['BPT', 'BPT+SAM_c']
