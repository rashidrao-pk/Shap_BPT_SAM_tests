import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from auc_export import AUC_VARIANTS, auc_columns, enrich_auc_rows
from export_auc_variants import export


def fixture():
    curve = {key: index / 10 for index, key in enumerate(AUC_VARIANTS)}
    curve.update(xs=[0., 1.], ys=[1., 0.], requested_points=200, evaluated_points=2,
                 flipped=True, f_0=.2, f_S=.8, score_delta_denominator=.6, predicted_cls=2)
    return {'label': 'BPT', 'auc_ins': dict(curve), 'auc_del': dict(curve)}


def test_all_variants_and_metadata_exported_without_curve_arrays():
    result = fixture()
    row = auc_columns(result)
    for direction in ['ins', 'del']:
        for key in AUC_VARIANTS:
            suffix = '_raw' if key == 'auc' else key[3:]
            assert row[f'auc_{direction}{suffix}'] == result[f'auc_{direction}'][key]
        assert row[f'auc_{direction}'] == result[f'auc_{direction}']['auc_clipr']
        assert row[f'auc_{direction}_evaluated_points'] == 2
        assert f'auc_{direction}_xs' not in row
    assert auc_columns({'method': 'no_auc'}) == {}


def test_csv_backfill_preserves_identifiers_and_makes_backup(tmp_path):
    run = tmp_path / 'run'; evaluation = tmp_path / 'eval'
    folder = run / '000000000139'; folder.mkdir(parents=True); evaluation.mkdir()
    (folder / '000000000139_metrics.json').write_text(json.dumps([fixture()]))
    name = 'auc_results_all_images_current_run.csv'
    original = 'image_id,method,time_exp\n000000000139,BPT,12.34\n'
    for directory in [run, evaluation]:
        (directory / name).write_text(original)
    export(run, evaluation)
    export(run, evaluation)
    for directory in [run, evaluation]:
        with (directory / name).open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0]['image_id'] == '000000000139'
        assert rows[0]['time_exp'] == '12.34'
        assert float(rows[0]['auc_del_adjr']) == fixture()['auc_del']['auc_adjr']
        assert (directory / (name + '.before_auc_variants.bak')).read_text() == original


def test_resumed_rows_enriched_by_method_and_unmatched_rows_preserved():
    rows = [{'method': 'BPT', 'time_exp': 3}, {'method': 'unsupported'}]
    enrich_auc_rows(rows, [fixture()])
    assert rows[0]['auc_ins_raw'] == 0
    assert rows[0]['time_exp'] == 3
    assert rows[1] == {'method': 'unsupported'}
