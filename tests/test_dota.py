"""DOTA: tiling, labels, ground truth, partition GT, the YOLO-OBB adapter, and runner registration.

No weights, GPU or network: the detector is a fake with Ultralytics' OBB result layout.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
sys.path.insert(0, str(ROOT / 'src/scripts/explanations'))
sys.path.insert(0, str(ROOT / 'src/hpc'))
import dota_data
from ground_truth import load_ground_truth


def box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


# ---------------------------------------------------------------- tiling

def source_scene(tmp_path):
    """A 2500 x 1500 source with objects at known places for tile=1024, stride=1024."""
    images, labels = tmp_path / 'raw/images', tmp_path / 'raw/labels'
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new('RGB', (2500, 1500), (40, 80, 120)).save(images / 'P0001.jpg')
    lines = ['imagesource:GoogleEarth', 'gsd:0.1',
             '100 100 300 100 300 200 100 200 plane 0',            # inside tile (0, 0)
             '100 400 200 400 200 500 100 500 small-vehicle 0',    # inside tile (0, 0)
             '600 600 700 600 700 700 600 700 storage-tank 0',     # inside tile (0, 0)
             '1000 100 1100 100 1100 200 1000 200 ship 0',         # 24% in (0, 0), 76% in (1024, 0)
             '1014 300 1114 300 1114 400 1014 400 harbor 0',       # 10% in (0, 0), 90% in (1024, 0)
             '200 800 260 800 260 860 200 860 helicopter 1',       # flagged difficult by DOTA
             '1500 1400 1600 1400 1600 1450 1500 1450 bridge 0']   # only in the bottom row
    (labels / 'P0001.txt').write_text('\n'.join(lines) + '\n')
    return images, labels


def test_windows_clamp_the_last_tile_inward_and_pad_small_sides():
    assert dota_data.window_starts(2500, 1024, 1024) == [0, 1024, 1476]
    assert dota_data.window_starts(1024, 1024, 1024) == [0]
    assert dota_data.window_starts(800, 1024, 1024) == [0]


def test_clip_and_area_are_exact_for_a_cut_box():
    square = box(0, 0, 10, 10)
    assert dota_data.polygon_area(square) == 100
    assert dota_data.polygon_area(dota_data.clip_polygon(square, 5, 0, 20, 20)) == pytest.approx(50)
    assert dota_data.polygon_area(dota_data.clip_polygon(square, 20, 20, 30, 30)) == 0
    rotated = np.array([[5, 0], [10, 5], [5, 10], [0, 5]], dtype=float)
    assert dota_data.polygon_area(dota_data.clip_polygon(rotated, 0, 0, 5, 10)) == pytest.approx(25)


def test_tiles_count_kept_objects_and_ignore_border_cut_and_difficult_ones(tmp_path):
    images, labels = source_scene(tmp_path)
    objects = dota_data.read_source_labels(labels / 'P0001.txt', 'dota-txt', 2500, 1500)
    plans = {plan['tile_id']: plan for plan in dota_data.plan_tiles('P0001', 2500, 1500, objects)}
    assert sorted(plans) == ['P0001__0_0', 'P0001__0_476', 'P0001__1024_0', 'P0001__1024_476',
                             'P0001__1476_0', 'P0001__1476_476']
    first = {item['class_name']: item for item in plans['P0001__0_0']['objects']}
    assert {name for name, item in first.items() if not item['difficult']} == {'plane', 'small vehicle', 'storage tank'}
    assert first['ship']['difficult'] and first['harbor']['difficult'] and first['helicopter']['difficult']
    second = {item['class_name']: item for item in plans['P0001__1024_0']['objects']}
    assert not second['ship']['difficult'] and not second['harbor']['difficult']
    # Corners move into tile coordinates without being clipped.
    np.testing.assert_allclose(second['ship']['polygon'], box(-24, 100, 76, 200))
    assert plans['P0001__0_0']['n_counted'] == 3 and plans['P0001__0_0']['n_ignored'] == 3


def test_prepare_tiles_writes_full_size_tiles_index_and_is_idempotent(tmp_path):
    images, labels = source_scene(tmp_path)
    output = tmp_path / 'prepared_1024/val'
    summary = dota_data.prepare_tiles(images, labels, 'dota-txt', output, min_objects=1, max_tiles=2, seed=0)
    assert summary['source_images'] == 1 and summary['tiles'] == 6 and summary['selected'] == 2
    tiles = sorted((output / 'images').glob('*.png'))
    assert len(tiles) == 2
    for tile in tiles:
        with Image.open(tile) as image:
            assert image.size == (1024, 1024)
        assert (output / 'labels' / f'{tile.stem}.txt').is_file()
    rows = dota_data.read_tile_index(output / 'tile_index.csv')
    assert sum(int(row['selected']) for row in rows) == 2 and len(rows) == summary['candidates']
    before = {path: path.stat().st_mtime_ns for path in output.rglob('*') if path.is_file()}
    again = dota_data.prepare_tiles(images, labels, 'dota-txt', output, min_objects=1, max_tiles=2, seed=0)
    assert again == summary
    assert {path: path.stat().st_mtime_ns for path in output.rglob('*') if path.is_file()} == before


def test_selection_is_seeded_and_sorted_independent():
    plans = [{'tile_id': f't{index:02d}', 'n_counted': 5} for index in range(20)]
    first = dota_data.select_tiles(plans, 3, 5, seed=1)[1]
    assert first == dota_data.select_tiles(list(reversed(plans)), 3, 5, seed=1)[1]
    assert first != dota_data.select_tiles(plans, 3, 5, seed=2)[1]


def test_bottom_padding_is_black(tmp_path):
    images, labels = tmp_path / 'i', tmp_path / 'l'
    images.mkdir()
    labels.mkdir()
    Image.new('RGB', (1024, 700), (200, 200, 200)).save(images / 'P0009.png')
    (labels / 'P0009.txt').write_text('10 10 50 10 50 50 10 50 plane 0\n')
    dota_data.prepare_tiles(images, labels, 'dota-txt', tmp_path / 'out', min_objects=1)
    pixels = np.asarray(Image.open(tmp_path / 'out/images/P0009__0_0.png'))
    assert pixels.shape == (1024, 1024, 3) and pixels[699].min() == 200 and pixels[700:].max() == 0


def test_normalised_ultralytics_labels_are_scaled_to_pixels(tmp_path):
    path = tmp_path / 'P0002.txt'
    path.write_text('2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n')
    (item,) = dota_data.read_source_labels(path, 'yolo-obb', 2000, 1000)
    assert item['class_name'] == 'storage tank' and item['difficult'] == 0
    np.testing.assert_allclose(item['polygon'], box(200, 100, 400, 200))


# ---------------------------------------------------------------- labels

def test_class_names_with_spaces_round_trip(tmp_path):
    objects = [{'polygon': box(1, 2, 3, 4), 'class_name': 'storage tank', 'difficult': 0},
               {'polygon': box(5, 6, 7, 8), 'class_name': 'ground track field', 'difficult': 1}]
    path = tmp_path / 'tile.txt'
    path.write_text(dota_data.format_tile_labels(objects))
    read = dota_data.read_tile_labels(path)
    assert [(item['class_name'], item['difficult']) for item in read] == [('storage tank', 0), ('ground track field', 1)]
    np.testing.assert_allclose(read[1]['polygon'], box(5, 6, 7, 8))


@pytest.mark.parametrize('line', ['1 2 3 4 5 6 7 8 plane', '1 2 3 4 5 6 7 x plane 0', '1 2 3 4 5 6 7 8 airship 0'])
def test_malformed_label_lines_name_the_file(tmp_path, line):
    path = tmp_path / 'broken_tile.txt'
    path.write_text(line + '\n')
    with pytest.raises(ValueError, match='broken_tile.txt'):
        dota_data.read_tile_labels(path)


# ---------------------------------------------------------------- ground truth

def tile_config(tmp_path, lines, image_size=64):
    images, labels = tmp_path / 'images', tmp_path / 'labels'
    images.mkdir()
    labels.mkdir()
    Image.new('RGB', (image_size, image_size)).save(images / 'P0001__0_0.png')
    (labels / 'P0001__0_0.txt').write_text('\n'.join(lines) + '\n')
    config = {'data': {'name': 'DOTA', 'input_format': 'directory', 'image_dir': str(images),
                       'ground_truth': {'format': 'dota', 'root': str(labels)}}}
    return config, images / 'P0001__0_0.png'


HARBOR_WITH_SHIP = ['0 0 40 0 40 40 0 40 harbor 0', '10 10 20 10 20 20 10 20 ship 0',
                    '50 50 60 50 60 60 50 60 ship 0', '30 50 40 50 40 60 30 60 ship 1']


def test_gt_union_of_one_class_and_of_all_classes(tmp_path):
    config, image = tile_config(tmp_path, HARBOR_WITH_SHIP)
    ship = load_ground_truth(config, image, (64, 64), gt_class='ship')
    assert ship['instance_count'] == 2 and ship['scope'] == 'selected_class'
    assert ship['mask'][15, 15] and ship['mask'][55, 55] and not ship['mask'][5, 5]
    everything = load_ground_truth(config, image, (64, 64))
    assert everything['instance_count'] == 3 and everything['mask'][5, 5]


def test_small_object_stays_visible_over_a_large_one(tmp_path):
    config, image = tile_config(tmp_path, HARBOR_WITH_SHIP)
    gt = load_ground_truth(config, image, (64, 64))
    assert gt['label_names'][int(gt['label_map'][15, 15])] == 'ship'
    assert gt['label_names'][int(gt['label_map'][5, 5])] == 'harbor'


def test_difficult_objects_are_ignored_pixels_not_mask(tmp_path):
    config, image = tile_config(tmp_path, HARBOR_WITH_SHIP)
    gt = load_ground_truth(config, image, (64, 64), gt_class='ship')
    assert not gt['mask'][55, 35] and gt['label_map'][55, 35] == dota_data.IGNORED_LABEL
    assert gt['ignored_pixels'] > 0 and gt['ignored_instances'] == 1


def test_class_names_are_normalised_and_unknown_names_raise(tmp_path):
    config, image = tile_config(tmp_path, ['1 1 9 1 9 9 1 9 storage tank 0'])
    assert load_ground_truth(config, image, (64, 64), gt_class='Storage-Tank')['instance_count'] == 1
    with pytest.raises(ValueError, match='Unknown DOTA class'):
        load_ground_truth(config, image, (64, 64), gt_class='airship')


def test_absent_class_gives_an_empty_mask_with_status(tmp_path):
    config, image = tile_config(tmp_path, ['1 1 9 1 9 9 1 9 plane 0'])
    gt = load_ground_truth(config, image, (64, 64), gt_class='bridge')
    assert not gt['mask'].any() and gt['status'] == 'no matching objects'


def test_box_outside_the_tile_raises_with_the_file(tmp_path):
    config, image = tile_config(tmp_path, ['900 900 950 900 950 950 900 950 plane 0'])
    with pytest.raises(ValueError, match='P0001__0_0.txt'):
        load_ground_truth(config, image, (64, 64))


def test_gt_metadata_is_json_serialisable(tmp_path):
    config, image = tile_config(tmp_path, HARBOR_WITH_SHIP)
    gt = load_ground_truth(config, image, (64, 64), gt_class='ship')
    json.dumps({key: value for key, value in gt.items() if key not in {'mask', 'label_map'}})


def test_partition_gt_gives_each_instance_in_full(tmp_path):
    from evaluate_partitions import gt_objects
    config, image = tile_config(tmp_path, ['0 0 30 0 30 30 0 30 plane 0', '20 20 40 20 40 40 20 40 plane 0',
                                           '50 0 60 0 60 10 50 10 ship 1'])
    objects = list(gt_objects(config, image, (64, 64)))
    assert [name for name, _, _ in objects] == ['plane', 'plane']
    first, second = objects[0][1], objects[1][1]
    assert first[25, 25] and second[25, 25]  # The overlap belongs to both.
    valid = objects[0][2]
    assert not valid[5, 55] and valid[25, 25]


# ---------------------------------------------------------------- adapter

def fake_result(classes, confidences, corners=None):
    """Ultralytics OBB result layout: boxes is None, detections in .obb (supports len)."""
    corners = corners if corners is not None else np.repeat(box(1, 1, 5, 5)[None], len(classes), axis=0)

    class Obb:
        cls, conf, xyxyxyxy = np.asarray(classes, float), np.asarray(confidences, float), np.asarray(corners, float)

        def __len__(self):
            return len(self.cls)
    return SimpleNamespace(boxes=None, obb=Obb())


class FakeObbModel:
    """Class 0 ('plane') confidence = mean of the channel Ultralytics reads as red (BGR index 2)."""
    task = 'obb'
    names = {index: name for index, name in enumerate(dota_data.CLASSES)}

    def __init__(self):
        self.received = []

    def predict(self, source, **options):
        self.received.extend(source)
        self.options = options
        results = []
        for image in source:
            red = float(np.asarray(image)[..., 2].mean()) / 255
            corners = np.array([[[2, 2], [10, 2], [10, 8], [2, 8]]], dtype=float)
            results.append(fake_result([0], [red], corners) if red > options['conf'] else fake_result([], []))
        return results


def adapter_args(**extra):
    return SimpleNamespace(model_batch_size=4, conf=.25, class_aggregation='sum', checkpoint=None,
                           model='yolo11s-obb', **extra)


def test_obb_scores_sum_or_max_and_tolerate_empty_and_unknown_classes():
    from model_adapters import obb_scores
    result = fake_result([1, 1, 3, 99], [.5, .25, .9, .8])
    assert obb_scores(result, 15, 'sum')[[1, 3]].tolist() == [.75, .9]
    assert obb_scores(result, 15, 'max')[1] == .5
    assert obb_scores(fake_result([], []), 15, 'sum').sum() == 0
    assert obb_scores(SimpleNamespace(obb=None, boxes=None), 15, 'sum').sum() == 0


def test_adapter_sends_bgr_to_ultralytics():
    from model_adapters import YoloObbAdapter
    model = FakeObbModel()
    adapter = YoloObbAdapter(adapter_args(), 'cpu', model=model)
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255  # RGB red
    scores = adapter.predict_batch([red])
    assert model.received[0][0, 0].tolist() == [0, 0, 255]  # BGR order reached the model
    assert scores[0, 0] == pytest.approx(1)
    assert model.options['imgsz'] == 1024 and model.options['half'] is False and model.options['conf'] == .25


def test_adapter_rejects_a_non_obb_model():
    from model_adapters import YoloObbAdapter
    model = FakeObbModel()
    model.task = 'detect'
    with pytest.raises(ValueError, match='obb'):
        YoloObbAdapter(adapter_args(), 'cpu', model=model)


def test_detections_carry_polygon_and_box():
    from model_adapters import YoloObbAdapter
    adapter = YoloObbAdapter(adapter_args(), 'cpu', model=FakeObbModel())
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[..., 0] = 255
    (found,) = adapter.detections(image)
    assert found['class_name'] == 'plane' and found['box'] == [2, 2, 10, 8] and len(found['polygon']) == 4


def test_cuda_device_is_passed_with_an_index():
    from model_adapters import yolo_device
    assert (yolo_device('cuda'), yolo_device('cuda:1'), yolo_device('cpu')) == ('cuda:0', 'cuda:1', 'cpu')


def test_weights_are_found_in_the_repository_checkpoint_folders(tmp_path):
    from model_adapters import resolve_yolo_weights
    (tmp_path / 'src/checkpoints').mkdir(parents=True)
    (tmp_path / 'src/checkpoints/yolo11m-obb.pt').write_bytes(b'x')
    assert resolve_yolo_weights(None, 'yolo11m-obb', tmp_path) == tmp_path / 'src/checkpoints/yolo11m-obb.pt'
    with pytest.raises(FileNotFoundError, match='dota'):
        resolve_yolo_weights(None, 'yolo11x-obb', tmp_path)


# ---------------------------------------------------------------- registration and runner

def test_config_and_runner_are_registered():
    import budget_ablation
    import check_partition_progress
    import experiment_common as common
    import prepare_full_budget_study as study
    import xai_leaderboard
    from summarize_ablation_study import DATASETS
    config = yaml.safe_load((ROOT / 'src/configs/DOTA_epito.yaml').read_text())
    assert config['data']['name'] == 'DOTA' and config['data']['ground_truth']['format'] == 'dota'
    assert config['data']['mask_dir_final'] == 'DOTA_sam2_tiny_62_compact-sam_200'
    assert config['model']['checkpoint'] == 'src/checkpoints/yolo11s-obb.pt'
    assert common.DEFAULT_CONFIGS['OD_DOTA'] == 'DOTA_epito' and common.MODEL_NAMES['OD_DOTA'] == 'yolo11s-obb'
    assert common.EXPECTED_DATASETS['OD_DOTA'] == 'dota' and 'OD_DOTA' in common.DETECTOR_RUNNERS
    assert 'OD_DOTA' in budget_ablation.RUNNERS and 'OD_DOTA' in DATASETS
    assert study.MODELS['DOTA'] == [('OD_DOTA', 'yolo11s-obb')] and 'DOTA' not in study.DEFAULT_DATASETS
    assert 'DOTA_epito' in check_partition_progress.DEFAULT_CONFIGS
    assert xai_leaderboard.DOMAINS['DOTA'] == 'remote sensing'
    args = common.create_parser('OD_DOTA').parse_args([])
    assert (args.config, args.conf, args.class_aggregation) == ('DOTA_epito', .25, 'sum')
    assert not hasattr(common.create_parser('OD_DETR').parse_args([]), 'conf')
    import generate_dota_partitions
    assert generate_dota_partitions.create_parser(default_config='DOTA_epito').parse_args([]).config == 'DOTA_epito'


def test_download_exposes_dota_pins_both_files_and_rejects_other_splits(tmp_path):
    import download_datasets
    assert 'dota' in download_datasets.DATASETS and len(download_datasets.DOTA_ZIP_SHA256) == 64
    ((url, path, digest),) = download_datasets.MODELS['dota']
    assert url.endswith('/yolo11s-obb.pt') and path == 'src/checkpoints/yolo11s-obb.pt' and len(digest) == 64
    for split in ['test', 'train']:
        with pytest.raises(ValueError, match='val only'):
            download_datasets.main(['--dataset', 'dota', '--assets', 'dataset', '--split', split,
                                    '--root', str(tmp_path)])
    assert not (tmp_path / 'DOTA').exists()


def run_fixture(tmp_path, red=255):
    """Two prepared tiles, compact partitions, labels and a config, in the layout the runner expects."""
    images, labels, parts = tmp_path / 'images', tmp_path / 'labels', tmp_path / 'parts/DOTA_test'
    for folder in [images, labels, parts]:
        folder.mkdir(parents=True)
    for tile, value in [('P0001__0_0', red), ('P0002__0_0', 0)]:
        pixels = np.zeros((16, 20, 3), dtype=np.uint8)
        pixels[:, 10:, 0] = value  # Red only on the right half, where the plane is.
        Image.fromarray(pixels).save(images / f'{tile}.png')
        (labels / f'{tile}.txt').write_text('10 0 20 0 20 16 10 16 plane 0\n')
        partition = np.zeros((16, 20), dtype=np.uint16)
        partition[:, 10:] = 1
        partition[8:, :10] = 2
        np.save(parts / f'{tile}_compact.npy', partition)
        (parts / f'{tile}_results.json').write_text(json.dumps({'saved_partitions': {'compact': 'x'}}))
    (tmp_path / 'weights.pt').write_bytes(b'x')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'data': {'name': 'DOTA', 'input_format': 'directory', 'image_dir': str(images), 'mask_dir_final': parts.name,
                 'ground_truth': {'format': 'dota', 'root': str(labels)}},
        'model': {'checkpoint': str(tmp_path / 'weights.pt')}, 'device': 'cpu',
        'output': {'partitions': str(parts.parent), 'explanations': str(tmp_path / 'e'),
                   'evaluations': str(tmp_path / 'v')}}))
    return config


def test_dry_run_validates_without_loading_a_model(tmp_path, monkeypatch):
    import experiment_common as common
    config = run_fixture(tmp_path)
    monkeypatch.setattr(common, 'load_adapter', lambda *a: pytest.fail('model loaded in a dry run'))
    args = common.create_parser('OD_DOTA').parse_args(['--config', str(config), '--show-gt', '--dry-run'])
    inputs = common.run('OD_DOTA', args)
    assert [record['image_id'] for record in inputs['records']] == ['P0001__0_0', 'P0002__0_0']


def test_explanation_run_scores_the_plane_and_skips_a_tile_without_detections(tmp_path):
    import csv
    import experiment_common as common
    from model_adapters import YoloObbAdapter
    config = run_fixture(tmp_path)
    args = common.create_parser('OD_DOTA').parse_args([
        '--config', str(config), '--show-gt', '--max-evals', '8', '--kernel-shap-samples', '10',
        '--auc-num-samples', '5', '--road-short-side', '0', '--no-heatmaps', '--verbose-level', 'low'])
    result = common.run('OD_DOTA', args, adapter=YoloObbAdapter(args, 'cpu', model=FakeObbModel()))
    assert result['completed'] == 1
    assert result['skipped'] == [{'image_id': 'P0002__0_0', 'reason': 'no detections above conf 0.25'}]
    folder = result['run_dir'] / 'P0001__0_0'
    prediction = json.loads((folder / 'P0001__0_0_predictions.json').read_text())
    assert prediction['explained_class'] == 'plane' and prediction['detections'][0]['polygon']
    gt = json.loads((folder / 'P0001__0_0_gt.json').read_text())
    assert gt['class_name'] == 'plane' and gt['instance_count'] == 1
    table = list(csv.DictReader((result['evaluation_dir'] / 'auc_results_all_images_current_run.csv').open()))
    assert {row['method'] for row in table} == set(common.METHODS)
    for row in table:
        assert row['auc_ins'] and row['road_gap'] and row['best_iou'] and row['pointing_accuracy']
    attributions = np.load(folder / 'P0001__0_0_attributions.npz')['SAM_c'][0]
    assert attributions[:, 10:].mean() > attributions[:, :10].mean()  # Credit goes to the plane's half.
    assert json.loads((result['run_dir'] / 'skipped.json').read_text())[0]['image_id'] == 'P0002__0_0'
    assert (result['evaluation_dir'] / 'winner.md').is_file()
    info = json.loads((result['run_dir'] / 'run_info.json').read_text())
    assert info['score_type'] == 'detection confidence'


def test_other_obb_weights_are_looked_up_by_name_not_the_config_file(tmp_path):
    import experiment_common as common
    config = run_fixture(tmp_path)
    args = common.create_parser('OD_DOTA').parse_args(['--config', str(config), '--model', 'yolo11m-obb'])
    common.prepare_inputs('OD_DOTA', args)
    assert args.checkpoint is None and args.model == 'yolo11m-obb'
    default = common.create_parser('OD_DOTA').parse_args(['--config', str(config)])
    common.prepare_inputs('OD_DOTA', default)
    assert default.checkpoint == tmp_path / 'weights.pt'


def test_summary_setup_counts_dota_labels(tmp_path):
    import summary_setup
    (tmp_path / 'labels').mkdir()
    for stem in ['P0001__0_0', 'P0001__1024_0']:
        (tmp_path / 'labels' / f'{stem}.txt').write_text('')
    data = {'name': 'DOTA', 'ground_truth': {'format': 'dota', 'root': str(tmp_path / 'labels')}}
    assert summary_setup.masks_found(data) == {'P0001__0_0', 'P0001__1024_0'}
    assert summary_setup.DATASETS['dota'].models == ('yolo_obb',)


def test_best_iou_leaves_dota_ignored_regions_out(tmp_path):
    from all_methods import best_iou
    config, image = tile_config(tmp_path, ['0 0 10 0 10 64 0 64 plane 0', '10 0 20 0 20 64 10 64 plane 1'])
    gt = load_ground_truth(config, image, (64, 64), gt_class='plane')
    heatmap = np.zeros((64, 64))
    heatmap[:, :21] = 1  # Covers the plane and the ignored half-plane.
    assert best_iou(heatmap, gt)['best_iou'] > .9
