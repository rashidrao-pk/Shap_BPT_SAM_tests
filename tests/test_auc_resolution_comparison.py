"""Paired AUC resolution tests without expensive explanation recomputation."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'))
import compare_auc_points as comparison
from experiment_common import load_runtime


def test_tied_masks_do_not_double_model_evaluations():
    runtime = load_runtime()
    def predict(masks):
        return masks.mean(axis=(1, 2))[:, None]
    rows, curves = comparison.compare_map(runtime.utx.saliency_to_auc, predict,
        np.zeros((10, 10), dtype=float), 0, 1., 0., repeats=2, reference=300)
    assert len(rows) == 4
    assert all(row['actual_points'] == 2 and row['mask_evaluations'] == 2 for row in rows)
    rows = [{**row, 'image_id': 'one', 'method': 'tied'} for row in rows]
    summary, _ = comparison.paired_summary(rows, [100, 200], .005)
    assert summary['max_absolute_auc_change'] == 0
    assert 'Use 100 points' in summary['recommendation']
    assert curves['ins_reference']['requested_points'] == 300


def test_recommendation_uses_reference_error_not_larger_auc():
    rows = []
    for n, auc, error in [(100, .8, .1), (200, .702, .002)]:
        rows.append({'image_id': 'one', 'method': 'one', 'direction': 'ins',
            'requested_points': n, 'auc': auc, 'absolute_reference_error': error,
            'median_seconds': n/100, 'mask_evaluations': n,
            'auc_repeat_range': 0, 'reference_points': 1000})
    summary, _ = comparison.paired_summary(rows, [100, 200], .005)
    assert 'Use 200 points' in summary['recommendation']
    assert summary['median_runtime_ratio'] == 2
    rows[1]['absolute_reference_error'] = .02
    summary, _ = comparison.paired_summary(rows, [100, 200], .005)
    assert 'No blanket recommendation' in summary['recommendation']


def test_saved_maps_comparison_does_not_run_explainer(tmp_path):
    source = tmp_path / 'run'
    folder = source / 'image1'
    folder.mkdir(parents=True)
    image = np.full((8, 12, 3), 100, dtype=np.uint8)
    image_path = tmp_path / 'image.png'
    Image.fromarray(image).save(image_path)
    (source / 'run_info.json').write_text(json.dumps({'runner': 'OC_ResNet', 'model': 'resnet50',
        'background_replacement_values': 'black', 'settings': {'model_batch_size': 2, 'auc_batch_size': 2}}))
    (folder / 'image1_predictions.json').write_text(json.dumps({'source_image': str(image_path),
        'predicted_class_id': 0, 'f_S': 100/255, 'f_0': 0}))
    np.savez_compressed(folder / 'image1_attributions.npz', BPT=np.arange(96, dtype=float).reshape(1, 8, 12))
    class Adapter:
        class_names = ['object', 'other']
        def predict_batch(self, images):
            values = np.asarray([np.asarray(image).mean()/255 for image in images])
            return np.column_stack([values, 1-values])
    original = (folder / 'image1_attributions.npz').read_bytes()
    args = comparison.parse_args(['--run-dir', str(source), '--points', '4', '8',
        '--reference-points', '16', '--repeats', '2', '--device', 'cpu', '--torch-threads', '1'])
    summary = comparison.run(args, adapter=Adapter())
    output = source / 'auc_point_comparison'
    for name in ['measurements.csv', 'paired_comparisons.csv', 'summary.json', 'conclusion.md', 'auc_resolution_comparison.png']:
        assert (output / name).is_file()
    assert summary['image_count'] == 1
    assert json.loads((output / 'comparison_status.json').read_text())['state'] == 'complete'
    assert (folder / 'image1_attributions.npz').read_bytes() == original


def test_legacy_yolo_restores_original_scoring_and_checkpoint():
    runner, args = comparison.restore_model_args({'model_group': 'yolov11', 'model': 'yolo11s',
        'model_source': 'src/checkpoints/yolo11s.pt', 'class_aggregation': 'max',
        'background_replacement_values': 'noise', 'auc_batch_size': 16}, 'cpu')
    assert runner == 'OD_yolo' and args.model == 'yolo11s'
    assert args.class_aggregation == 'max' and args.bg_type == 'noise'
    assert args.auc_batch_size == 16 and args.checkpoint.is_absolute()


def test_changed_method_ranking_blocks_blanket_low_resolution_choice():
    rows = []
    for n, scores in [(100, [.500, .501]), (200, [.502, .500])]:
        for method, auc in zip(['one', 'two'], scores):
            rows.append({'image_id': 'one', 'method': method, 'direction': 'ins',
                'requested_points': n, 'auc': auc, 'absolute_reference_error': .001,
                'median_seconds': n/100, 'mask_evaluations': n,
                'auc_repeat_range': 0, 'reference_points': 1000})
    summary, _ = comparison.paired_summary(rows, [100, 200], .005)
    assert summary['ranking_changed_image_directions'] == 1
    assert 'No blanket recommendation' in summary['recommendation']
