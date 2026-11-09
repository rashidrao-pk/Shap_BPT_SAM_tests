"""Who-wins tables for one run and across datasets."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
sys.path.insert(0, str(ROOT / 'src/scripts/explanations'))
import xai_leaderboard as board

METHODS = ['BPT-1000', 'LIME-1000', 'GradCAM']


def run_frame(images=20, quality=(.8, .5, .3), seed=0, road=True, images_prefix='img'):
    """Per image, a method with higher quality has higher insertion, lower deletion, larger ROAD gap."""
    rng = np.random.default_rng(seed)
    rows = []
    for image in range(images):
        confidence = rng.uniform(.2, 1)
        for method, level in zip(METHODS, quality):
            noise = rng.normal(0, .05)
            row = {'image_id': f'{images_prefix}{image:03d}', 'method': method,
                   'auc_ins': confidence * (level + noise), 'auc_del': confidence * (1 - level + noise),
                   'time_exp': 1 + level}
            if road:
                row.update({'road_morf': confidence * (1 - level), 'road_lerf': confidence,
                            'road_gap': confidence * level, 'road_morf_rel': 1 - level, 'road_lerf_rel': 1.,
                            'road_gap_rel': level})
            rows.append(row)
    return pd.DataFrame(rows)


def test_best_method_wins_every_metric_and_overall():
    result, winners, used = board.leaderboard(board.use_relative_road(run_frame()))
    assert list(result['method']) == METHODS
    assert used == ['auc_ins', 'auc_del', 'road_gap']
    winner = winners.set_index('metric')['winner']
    assert set(winner.drop('time_exp')) == {'BPT-1000'}
    assert winner['time_exp'] == 'GradCAM'  # Cost is ranked too: the fastest wins it.
    faithfulness = winners[winners['metric'].isin(used)]
    assert faithfulness['clear_winner'].all()
    assert result.loc[0, 'auc_ins_rank'] == 1 and result.loc[0, 'auc_ins_win_share'] == 1


def test_lower_is_better_metrics_rank_the_smallest_first():
    frame = pd.DataFrame({'image_id': ['a', 'a', 'b', 'b'], 'method': ['X', 'Y'] * 2, 'auc_del': [.1, .9, .2, .8]})
    result, winners, _ = board.leaderboard(frame)
    assert winners.set_index('metric').loc['auc_del', 'winner'] == 'X'
    assert result.set_index('method').loc['X', 'auc_del_rank'] == 1


def test_equal_methods_are_not_called_a_clear_win():
    frame = run_frame(quality=(.5, .5, .5))
    _, winners, _ = board.leaderboard(frame)
    assert not winners.set_index('metric').loc['auc_ins', 'clear_winner']


def test_relative_road_is_used_only_when_every_image_has_it():
    frame = run_frame()
    assert board.use_relative_road(frame)['road_gap'].equals(frame['road_gap_rel'])
    frame.loc[0, 'road_gap_rel'] = np.nan  # e.g. a negative logit
    assert board.use_relative_road(frame)['road_gap'].equals(frame['road_gap'])


def test_images_missing_a_method_are_left_out_of_the_paired_ranks():
    frame = run_frame(images=4)
    frame = frame[~((frame['image_id'] == 'img000') & (frame['method'] == 'GradCAM'))]
    assert len(board.paired(frame, 'auc_ins')) == 3


def test_dataset_report_writes_tables_plots_and_a_preview_warning(tmp_path):
    result = board.dataset_report(run_frame(images=4), tmp_path, 'demo')
    for name in ['leaderboard.csv', 'metric_winners.csv', 'winner.md', 'method_boxplots.png', 'method_ranks.png']:
        assert (tmp_path / name).is_file(), name
    text = (tmp_path / 'winner.md').read_text()
    assert 'BPT-1000' in text and 'Preview only' in text
    assert result['overall_metrics'] == ['auc_ins', 'auc_del', 'road_gap']


def test_run_without_auc_still_ranks_on_road():
    frame = run_frame().drop(columns=['auc_ins', 'auc_del'])
    result, _, used = board.leaderboard(frame)
    assert used == ['road_gap'] and result.loc[0, 'method'] == 'BPT-1000'


def write_run(root, runner, partitions, frame, folder='bg_full_B1000'):
    path = root / runner / partitions / folder
    path.mkdir(parents=True)
    frame.to_csv(path / board.TABLE, index=False)


def test_all_report_ranks_across_datasets_and_domains(tmp_path):
    write_run(tmp_path, 'COD_ViT', 'CAMO_sam2_tiny_62_compact-sam_50', run_frame(seed=1))
    write_run(tmp_path, 'OC_BUSI', 'BUSI_sam2_tiny_62_compact-sam_50', run_frame(seed=2, quality=(.4, .8, .2)))
    write_run(tmp_path, 'OC_ResNet', 'ImageNet_sam2_tiny_62_compact-sam_100', run_frame(seed=3))
    # Two runs of one dataset count once.
    write_run(tmp_path, 'OD_DETR', 'MSCOCO_sam1_vit_b_62_compact-sam_200', run_frame(seed=4))
    write_run(tmp_path, 'OD_yolo', 'MSCOCO_sam1_vit_b_62_compact-sam_200', run_frame(seed=5))
    # Other budgets, and tables outside bg_*_B*, are ignored.
    write_run(tmp_path, 'COD_ViT', 'CAMO_sam2_tiny_62_compact-sam_50', run_frame(seed=6, quality=(.1, .2, .9)),
              'bg_full_B500')
    (tmp_path / 'budget_steps').mkdir()
    run_frame().to_csv(tmp_path / 'budget_steps' / board.TABLE, index=False)
    output = tmp_path / 'out'
    result = board.all_report(tmp_path, output, budget=1000)
    overall = result['overall']
    assert overall.loc[0, 'method'] == 'BPT-1000'
    assert overall.set_index('method').loc['BPT-1000', 'datasets_won'] == 3
    assert overall.set_index('method').loc['LIME-1000', 'domains_won'] == 1  # medical
    assert len(result['datasets']) == 4
    assert set(result['domains'].index) == {'camouflage', 'medical', 'natural images'}
    for name in ['runs.csv', 'run_ranks.csv', 'dataset_ranks.csv', 'domain_ranks.csv', 'overall.csv', 'winner.md',
                 'dataset_ranks.png', 'domain_ranks.png', 'rank_boxplots.png']:
        assert (output / name).is_file(), name
    assert '**medical:** LIME-1000' in (output / 'winner.md').read_text()


def test_runs_missing_a_shared_method_are_left_out_and_listed(tmp_path):
    for index, dataset in enumerate(['CAMO', 'BUSI', 'ImageNet']):
        write_run(tmp_path, f'R{index}', f'{dataset}_sam2_tiny_62_compact-sam_50', run_frame(seed=index))
    write_run(tmp_path, 'OC_PACO', 'PACO_sam2_tiny_62_compact-sam_50',
              run_frame()[lambda frame: frame['method'] != 'GradCAM'])
    result = board.all_report(tmp_path, tmp_path / 'out')
    assert 'PACO' not in {dataset for _, dataset in result['datasets'].index}
    assert 'Left out 1 runs' in (tmp_path / 'out/winner.md').read_text()


def test_no_runs_names_the_expected_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match='bg_'):
        board.all_report(tmp_path, tmp_path / 'out')


def test_command_line_dataset_report(tmp_path, capsys):
    write_run(tmp_path, 'COD_ViT', 'CAMO_sam2_tiny_62_compact-sam_50', run_frame())
    folder = tmp_path / 'COD_ViT/CAMO_sam2_tiny_62_compact-sam_50/bg_full_B1000'
    board.main(['dataset', str(folder), '--no-plots'])
    assert 'Who wins' in capsys.readouterr().out
    assert (folder / 'leaderboard.csv').is_file() and not (folder / 'method_boxplots.png').exists()
