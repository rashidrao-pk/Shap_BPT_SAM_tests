"""ROAD imputation equations, equal removal across methods, score direction, and resume."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
sys.path.insert(0, str(ROOT / 'src/scripts/explanations'))
import evaluate_road
import road


def weighted_neighbours(image, y, x):
    total, weight_sum = np.zeros(image.shape[2]), 0
    for (dy, dx), weight in road.NEIGHBOURS:
        if 0 <= y + dy < image.shape[0] and 0 <= x + dx < image.shape[1]:
            total += weight * image[y + dy, x + dx]
            weight_sum += weight
    return total / weight_sum


@pytest.mark.parametrize('fraction', [.1, .5, .9])
def test_each_filled_pixel_is_the_weighted_mean_of_its_neighbours(fraction):
    rng = np.random.default_rng(0)
    image = rng.random((20, 24, 3))
    keep = rng.random((20, 24)) > fraction
    keep[0, 0] = True
    filled = road.impute(image, keep, noise=0)
    np.testing.assert_array_equal(filled[keep], image[keep])
    for y, x in zip(*np.nonzero(~keep)):
        np.testing.assert_allclose(filled[y, x], weighted_neighbours(filled, y, x), atol=1e-9)


def test_constant_image_is_restored_exactly():
    keep = np.ones((12, 12), dtype=bool)
    keep[3:9, 3:9] = False
    np.testing.assert_allclose(road.impute(np.full((12, 12, 3), .4), keep, noise=0), .4)


def test_large_hole_leaves_no_sharp_mask_outline():
    image = np.zeros((32, 32, 3))
    image[:, 16:] = 1
    keep = np.ones((32, 32), dtype=bool)
    keep[8:24, 8:24] = False
    filled = road.impute(image, keep, noise=0)
    assert np.abs(np.diff(filled[16, 7:25, 0])).max() < .2  # A black fill would jump by 1.


def test_noise_is_added_only_to_filled_pixels():
    image = np.full((8, 8, 3), .5)
    keep = np.ones((8, 8), dtype=bool)
    keep[2:5, 2:5] = False
    filled = road.impute(image, keep, noise=.01, rng=np.random.default_rng(0))
    assert (filled[keep] == .5).all() and filled[~keep].std() > 0


def test_impute_needs_a_kept_pixel():
    with pytest.raises(ValueError, match='kept pixel'):
        road.impute(np.zeros((4, 4, 3)), np.zeros((4, 4), dtype=bool))


@pytest.mark.parametrize('direction', road.ORDERS)
def test_tied_and_pixel_level_maps_remove_the_same_count(direction):
    rng = np.random.default_rng(0)
    tie = rng.random((16, 16))
    pixel_map = rng.random((16, 16))
    region_map = np.repeat(np.repeat(rng.random((2, 2)), 8, axis=0), 8, axis=1)
    for fraction in road.FRACTIONS:
        counts = {(~road.keep_mask(road.removal_order(values, tie), fraction, values.shape, direction)).sum()
                  for values in [pixel_map, region_map, np.zeros((16, 16))]}
        assert counts == {round(fraction * 256)}


def test_morf_removes_high_and_lerf_removes_low_attribution():
    heatmap = np.arange(100.).reshape(10, 10)
    order = road.removal_order(heatmap, np.zeros((10, 10)))
    assert not road.keep_mask(order, .1, (10, 10), 'morf')[9, 9]
    assert road.keep_mask(order, .1, (10, 10), 'morf')[0, 0]
    assert not road.keep_mask(order, .1, (10, 10), 'lerf')[0, 0]
    assert road.keep_mask(order, .1, (10, 10), 'lerf')[9, 9]


def bright_patch():
    """The model's score is the patch brightness, so a faithful map marks the patch."""
    image = np.full((32, 32, 3), 40, dtype=np.uint8)
    image[4:12, 4:12] = 255
    truth = np.zeros((32, 32))
    truth[4:12, 4:12] = 1

    def predict(images):
        score = np.array([np.asarray(item, dtype=float)[4:12, 4:12].mean() / 255 for item in images])
        return np.stack([1 - score, score], axis=1)
    return image, truth, predict


def test_faithful_map_beats_a_reversed_map():
    image, truth, predict = bright_patch()
    good = road.evaluate(predict, image, truth, 1, noise=0, short_side=0)
    bad = road.evaluate(predict, image, -truth, 1, noise=0, short_side=0)
    assert good['morf'] < bad['morf'] and good['lerf'] > bad['lerf'] and good['gap'] > 0 > bad['gap']


def test_evaluation_is_reproducible_and_shares_noise():
    image, truth, predict = bright_patch()
    first = road.evaluate(predict, image, truth, 1, seed=3, short_side=0)
    assert first == road.evaluate(predict, image, truth, 1, seed=3, short_side=0)


def test_area_is_the_mean_score():
    assert road.area([.5, 1], [1, 1], 1) == 1
    assert road.area([.5], [0], 1) == .5


def test_relative_scores_are_omitted_for_negative_logits():
    image, truth, _ = bright_patch()
    result = road.evaluate(lambda items: np.full((len(items), 1), -2.), image, truth, 0, fractions=[.5], short_side=0)
    assert result['morf'] == -2 and result['morf_rel'] is None


@pytest.mark.parametrize('fractions', [[], [0], [1], [.2, 1.5]])
def test_invalid_fractions_are_rejected(fractions):
    image, truth, predict = bright_patch()
    with pytest.raises(ValueError, match='fractions'):
        road.evaluate(predict, image, truth, 1, fractions=fractions)


def test_working_size_only_scales_down():
    assert road.working_size(1000, 2000, 256) == (256, 512)
    assert road.working_size(100, 200, 256) == (100, 200)
    assert road.working_size(1000, 2000, 0) == (1000, 2000)


def test_downscaling_keeps_small_attributed_regions():
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    heatmap = np.zeros((512, 512))
    heatmap[100, 100] = 16.
    pixels, values = road.resize_inputs(image, heatmap, 128)
    assert pixels.shape[:2] == values.shape == (128, 128) and values.max() > 0


class Adapter:
    def __init__(self):
        self.calls = 0

    def predict_batch(self, images):
        self.calls += len(images)
        return bright_patch()[2](images)


def run_folder(tmp_path):
    """A finished run on disk, in the layout written by experiment_common.run."""
    image, truth, _ = bright_patch()
    images, parts = tmp_path / 'images', tmp_path / 'parts/KvasirSEG_test'
    images.mkdir()
    parts.mkdir(parents=True)
    Image.fromarray(image).save(images / 'a.png')
    np.save(parts / 'a_compact.npy', np.zeros((32, 32), dtype=np.int32))
    (parts / 'a_results.json').write_text(json.dumps({'saved_partitions': {'compact': 'x'}}))
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'data': {'name': 'KvasirSEG', 'input_format': 'directory', 'image_dir': str(images),
                 'mask_dir_final': parts.name},
        'model': {'checkpoint': str(checkpoint)},
        'output': {'partitions': str(parts.parent), 'explanations': str(tmp_path / 'e'),
                   'evaluations': str(tmp_path / 'v')}}))
    folder = tmp_path / 'e/OC_KvasirSEG' / parts.name / 'bg_full_B100/a'
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'a_attributions.npz', Good=truth[None], Reversed=-truth[None])
    (folder / 'a_predictions.json').write_text(json.dumps({'predicted_class_id': 1}))
    return config, folder


def arguments(config, *extra):
    argv = ['--runner', 'OC_KvasirSEG', '--config', str(config), '--budget', '100', '--road-short-side', '0', *extra]
    return evaluate_road.parser(argv).parse_args(argv)


def test_finished_run_is_scored_from_saved_attributions(tmp_path):
    config, folder = run_folder(tmp_path)
    summary = evaluate_road.run(arguments(config), adapter=Adapter())
    assert [item['method'] for item in summary] == ['Good', 'Reversed']
    assert summary[0]['gap_mean'] > summary[1]['gap_mean']
    assert (folder / 'a_road.json').is_file()
    table = tmp_path / 'v/OC_KvasirSEG/KvasirSEG_test/bg_full_B100'
    assert len(list(csv.DictReader((table / 'road_results.csv').open()))) == 2
    assert (table / 'road_summary.csv').is_file()


def test_repeat_run_resumes_but_new_settings_recompute(tmp_path):
    config, _ = run_folder(tmp_path)
    evaluate_road.run(arguments(config), adapter=Adapter())
    repeat = Adapter()
    evaluate_road.run(arguments(config), adapter=repeat)
    assert repeat.calls == 0
    changed = Adapter()
    evaluate_road.run(arguments(config, '--road-seed', '5'), adapter=changed)
    assert changed.calls > 0


def test_methods_filter_selects_saved_maps(tmp_path):
    config, _ = run_folder(tmp_path)
    summary = evaluate_road.run(arguments(config, '--methods', 'Good'), adapter=Adapter())
    assert [item['method'] for item in summary] == ['Good']


def test_missing_run_names_the_requirement(tmp_path):
    config, folder = run_folder(tmp_path)
    with pytest.raises(FileNotFoundError, match='same --budget'):
        evaluate_road.run(evaluate_road.parser(['--runner', 'OC_KvasirSEG', '--config', str(config), '--budget', '999'])
                          .parse_args(['--runner', 'OC_KvasirSEG', '--config', str(config), '--budget', '999']), Adapter())


def test_attribution_shape_mismatch_is_rejected(tmp_path):
    config, folder = run_folder(tmp_path)
    np.savez_compressed(folder / 'a_attributions.npz', Bad=np.zeros((1, 8, 8)))
    with pytest.raises(ValueError, match='does not match image'):
        evaluate_road.run(arguments(config), adapter=Adapter())


def test_image_seed_is_stable():
    assert evaluate_road.image_seed(0, 'abc') == evaluate_road.image_seed(0, 'abc') != evaluate_road.image_seed(0, 'acb')
