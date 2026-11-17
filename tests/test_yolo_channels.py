"""YOLO must see the same colours whether it reads a file itself or gets our RGB arrays (T-0a)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src/scripts/explanations'), str(ROOT / 'src/scripts')]
from yolo_io import YOLO_INPUT_VERSION, to_bgr, ultralytics_input  # noqa: E402

WEIGHTS = next((path for path in [ROOT / 'checkpoints/yolo11s.pt', ROOT / 'src/checkpoints/yolo11s.pt'] if path.is_file()), None)


def test_rgb_arrays_are_reversed_and_other_inputs_pass_through():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # red
    bgr = to_bgr(rgb)
    assert bgr[..., 2].max() == 200 and bgr[..., 0].max() == 0 and bgr.flags['C_CONTIGUOUS']
    assert ultralytics_input('image.jpg') == 'image.jpg'
    assert all(item[..., 2].max() == 200 for item in ultralytics_input([rgb, rgb]))
    assert all(item[..., 2].max() == 200 for item in ultralytics_input(np.stack([rgb, rgb])))


def test_budget_steps_signature_carries_the_input_version():
    import budget_steps
    assert budget_steps.YOLO_INPUT_VERSION == YOLO_INPUT_VERSION == 2


def test_old_result_tables_are_set_aside_once(tmp_path):
    import OD_yolo
    (tmp_path / 'auc_results_all_images_current_run.csv').write_text('image_no,method\n1,BPT\n')
    assert OD_yolo.retire_old_yolo_results(tmp_path) == 1
    assert not (tmp_path / 'auc_results_all_images_current_run.csv').exists()
    assert (tmp_path / 'auc_results_all_images_current_run.v1_swapped_channels.csv').is_file()
    assert json.loads((tmp_path / 'yolo_input.json').read_text())['yolo_input_version'] == YOLO_INPUT_VERSION
    (tmp_path / 'auc_results_all_images_current_run.csv').write_text('image_no,method\n1,BPT\n')  # A new run's table.
    assert OD_yolo.retire_old_yolo_results(tmp_path) == 0
    assert (tmp_path / 'auc_results_all_images_current_run.csv').is_file()


@pytest.fixture(scope='module')
def red_image(tmp_path_factory):
    from PIL import Image
    rgb = np.asarray(Image.open(ROOT / 'data/sample_images/truck.jpg').convert('RGB')).astype(float)
    rgb[..., 0] = np.clip(rgb[..., 0] * 1.6 + 60, 0, 255)  # Strongly red, so a swap changes the detections.
    rgb[..., 2] *= .4
    rgb = rgb.astype(np.uint8)
    path = tmp_path_factory.mktemp('yolo') / 'red_truck.png'  # Lossless: the file decodes to exactly rgb.
    Image.fromarray(rgb).save(path)
    return rgb, path


@pytest.fixture(scope='module')
def yolo():
    if WEIGHTS is None:
        pytest.skip('yolo11s.pt not found in checkpoints/ or src/checkpoints/')
    pytest.importorskip('ultralytics')
    import OD_yolo
    OD_yolo.load_runtime_dependencies()
    return OD_yolo, OD_yolo.YOLO(str(WEIGHTS))


def test_runner_and_budget_adapter_match_ultralytics_reading_the_file(yolo, red_image):
    od, model = yolo
    rgb, path = red_image
    classes = od.yolo_class_count(model)
    expected = od.yolo_result_to_scores(model.predict(source=str(path), verbose=False)[0], classes)
    assert expected.sum() > 0
    np.testing.assert_allclose(od.predict_yolo(model, rgb, num_classes=classes), expected, atol=1e-5)
    np.testing.assert_allclose(od.predict_yolo_batch(model, [rgb], num_classes=classes)[0], expected, atol=1e-5)

    import budget_ablation
    adapter = object.__new__(budget_ablation.YoloAdapter)  # Skip weight resolution; use the loaded model.
    adapter.od, adapter.model, adapter.batch_size, adapter.aggregate, adapter.device = od, model, 4, 'sum', 'cpu'
    adapter.class_names = [model.names[i] for i in range(classes)]
    np.testing.assert_allclose(adapter.predict_batch([rgb])[0], expected, atol=1e-5)

    swapped = od.yolo_result_to_scores(model.predict(source=rgb, verbose=False)[0], classes)  # The old behaviour.
    assert not np.allclose(swapped, expected, atol=1e-3), 'the fixture must be sensitive to the channel order'
