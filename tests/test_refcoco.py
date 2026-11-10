"""RefCOCO + CLIP: safe refs loading, expressions, ground truth, the CLIP adapter, and the VL_RefCOCO runner.

No weights or network: CLIP is replaced by a small fake with the same methods.
"""
import json
from pathlib import Path
import pickle
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
import refcoco_data

STEMS = ['COCO_train2014_000000000001', 'COCO_train2014_000000000002']


def square(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def instances():
    images = [{'id': index + 1, 'file_name': f'{stem}.jpg', 'height': 16, 'width': 20} for index, stem in enumerate(STEMS)]
    annotations = [
        {'id': 11, 'image_id': 1, 'category_id': 1, 'segmentation': square(10, 0, 19, 15), 'area': 150, 'iscrowd': 0, 'bbox': [10, 0, 10, 16]},
        {'id': 12, 'image_id': 1, 'category_id': 1, 'segmentation': square(0, 0, 9, 15), 'area': 150, 'iscrowd': 0, 'bbox': [0, 0, 10, 16]},
        {'id': 13, 'image_id': 1, 'category_id': 2, 'segmentation': square(0, 0, 3, 3), 'area': 16, 'iscrowd': 0, 'bbox': [0, 0, 4, 4]},
        {'id': 21, 'image_id': 2, 'category_id': 2, 'segmentation': square(10, 0, 19, 15), 'area': 150, 'iscrowd': 0, 'bbox': [10, 0, 10, 16]}]
    return {'images': images, 'annotations': annotations,
            'categories': [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'dog'}]}


def refs():
    def sentence(sent_id, text):
        return {'sent_id': sent_id, 'sent': text, 'raw': text.upper(), 'tokens': text.split()}
    return [
        {'ref_id': 0, 'ann_id': 11, 'image_id': 1, 'split': 'val', 'category_id': 1, 'file_name': f'{STEMS[0]}_11.jpg',
         'sent_ids': [0, 1], 'sentences': [sentence(0, 'red man right'), sentence(1, 'man on the right')]},
        {'ref_id': 1, 'ann_id': 12, 'image_id': 1, 'split': 'val', 'category_id': 1, 'file_name': f'{STEMS[0]}_12.jpg',
         'sent_ids': [2], 'sentences': [sentence(2, 'man on the left')]},
        {'ref_id': 2, 'ann_id': 21, 'image_id': 2, 'split': 'val', 'category_id': 2, 'file_name': f'{STEMS[1]}_21.jpg',
         'sent_ids': [3], 'sentences': [sentence(3, 'red dog')]},
        {'ref_id': 3, 'ann_id': 13, 'image_id': 1, 'split': 'train', 'category_id': 2, 'file_name': f'{STEMS[0]}_13.jpg',
         'sent_ids': [4], 'sentences': [sentence(4, 'small dog')]}]


# ---------------------------------------------------------------- data

def test_refs_pickle_loads_plain_data_only():
    assert refcoco_data.read_refs(pickle.dumps(refs(), protocol=2))[0]['ref_id'] == 0

    class Payload:
        def __reduce__(self):
            return (print, ('this must never run',))
    with pytest.raises(pickle.UnpicklingError, match='only plain data'):
        refcoco_data.read_refs(pickle.dumps([Payload()]))


def test_expressions_cover_every_sentence_of_the_split():
    entries = refcoco_data.build_expressions(refs(), instances(), 'val')
    assert [entry['record_id'] for entry in entries] == [
        f'{STEMS[0]}__ref0_s0', f'{STEMS[0]}__ref0_s1', f'{STEMS[0]}__ref1_s2', f'{STEMS[1]}__ref2_s3']
    first = entries[0]
    assert (first['image_stem'], first['ann_id'], first['category'], first['expression']) == (STEMS[0], 11, 'person', 'red man right')
    assert len(refcoco_data.select_expressions(entries)) == 3
    assert len(refcoco_data.select_expressions(entries, all_sentences=True)) == 4
    with pytest.raises(ValueError, match='--split val'):
        refcoco_data.build_expressions(refs(), instances(), 'testA')


def test_missing_annotation_is_reported():
    broken = instances()
    broken['annotations'] = broken['annotations'][1:]
    with pytest.raises(ValueError, match='annotation 11'):
        refcoco_data.build_expressions(refs(), broken, 'val')


def dataset(tmp_path):
    """Two images, instances.json, expressions.json, compact partitions and a config, as prepared."""
    raw, prepared, parts = tmp_path / 'raw/refcoco', tmp_path / 'prepared/val', tmp_path / 'parts/RefCOCO_test'
    for folder in [raw, prepared / 'images', parts]:
        folder.mkdir(parents=True)
    (raw / 'instances.json').write_text(json.dumps(instances()))
    entries = refcoco_data.build_expressions(refs(), instances(), 'val')
    refcoco_data.write_expressions(prepared / 'expressions.json', entries, 'val', 'test')
    for stem in STEMS:
        pixels = np.zeros((16, 20, 3), dtype=np.uint8)
        pixels[:, 10:, 0] = 255  # The referred red object is on the right half.
        Image.fromarray(pixels).save(prepared / 'images' / f'{stem}.jpg', quality=100)
        partition = np.zeros((16, 20), dtype=np.uint16)
        partition[:, 10:] = 1
        partition[8:, :10] = 2
        np.save(parts / f'{stem}_compact.npy', partition)
        (parts / f'{stem}_results.json').write_text(json.dumps({'saved_partitions': {'compact': 'x'}}))
    (tmp_path / 'clip').mkdir()
    settings = {'format': 'refcoco', 'root': str(raw), 'annotation_file': str(raw / 'instances.json'),
                'expressions': str(prepared / 'expressions.json')}
    config = {'data': {'name': 'RefCOCO', 'input_format': 'directory', 'image_dir': str(prepared / 'images'),
                       'expressions': str(prepared / 'expressions.json'), 'mask_dir_final': parts.name,
                       'ground_truth': settings},
              'model': {'checkpoint': str(tmp_path / 'clip')}, 'device': 'cpu',
              'output': {'partitions': str(parts.parent), 'explanations': str(tmp_path / 'e'),
                         'evaluations': str(tmp_path / 'v')}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, config


# ---------------------------------------------------------------- ground truth

def test_gt_is_the_referred_object_only(tmp_path):
    from ground_truth import load_ground_truth
    _, config = dataset(tmp_path)
    image = Path(config['data']['image_dir']) / f'{STEMS[0]}.jpg'
    gt = load_ground_truth(config, image, (16, 20), gt_class='11')
    assert gt['scope'] == 'referred_object' and gt['class_name'] == 'person' and gt['instance_count'] == 1
    assert gt['mask'][5, 15] and not gt['mask'][5, 5]
    every = load_ground_truth(config, image, (16, 20))
    assert every['instance_count'] == 2 and every['mask'][5, 5] and every['mask'][5, 15]
    assert sorted(every['annotation_ids']) == [11, 12]  # The dog nobody referred to (13) is not GT.
    with pytest.raises(ValueError, match='not a referred object'):
        load_ground_truth(config, image, (16, 20), gt_class='13')
    json.dumps({key: value for key, value in gt.items() if key not in {'mask', 'label_map'}})


def test_partition_gt_lists_each_referred_object(tmp_path):
    from evaluate_partitions import gt_objects
    _, config = dataset(tmp_path)
    objects = list(gt_objects(config, Path(config['data']['image_dir']) / f'{STEMS[0]}.jpg', (16, 20)))
    assert sorted(name for name, _, _ in objects) == ['person', 'person']


# ---------------------------------------------------------------- adapter

class FakeTokenizer:
    def __call__(self, texts, **options):
        import torch
        return {'input_ids': torch.tensor([[1.0 if 'red' in texts[0] else -1.0]])}


class FakeClip:
    """Text 'red ...' embeds to +x; an image embeds to (red share, 0.5): cosine grows with red."""

    def __init__(self):
        import torch
        self.torch = torch
        self.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=4))

    def to(self, device):
        return self

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self

    def get_text_features(self, input_ids):
        return self.torch.cat([input_ids, self.torch.zeros_like(input_ids)], dim=1)

    def get_image_features(self, pixel_values):
        red = pixel_values[:, 0].mean(dim=(1, 2)) - pixel_values[:, 1].mean(dim=(1, 2)) + 3
        return self.torch.stack([red, self.torch.full_like(red, .5)], dim=1)


def fake_processor():
    return SimpleNamespace(tokenizer=FakeTokenizer(), image_processor=SimpleNamespace(
        crop_size={'height': 8, 'width': 8}, image_mean=[.5, .5, .5], image_std=[.5, .5, .5]))


def clip_adapter():
    from model_adapters import ClipAdapter
    args = SimpleNamespace(model_batch_size=4, checkpoint=None, model='openai/clip-vit-base-patch32', local_files_only=False)
    return ClipAdapter(args, 'cpu', model=FakeClip(), processor=fake_processor())


def test_clip_adapter_scores_cosine_similarity_with_the_current_expression():
    adapter = clip_adapter()
    red = np.zeros((16, 20, 3), dtype=np.uint8)
    red[..., 0] = 255
    green = np.zeros((16, 20, 3), dtype=np.uint8)
    green[..., 1] = 255
    with pytest.raises(RuntimeError, match='set_record'):
        adapter.predict_batch([red])
    adapter.set_record({'expression': 'red man'})
    scores = adapter.predict_batch([red, green])
    assert scores.shape == (2, 1) and adapter.class_names == ['red man']
    assert -1 <= scores.min() and scores.max() <= 1 and scores[0, 0] > scores[1, 0]
    adapter.set_record({'expression': 'blue man'})
    assert adapter.predict_batch([red])[0, 0] < 0  # Text now points the other way.


def test_clip_preprocessing_resizes_the_whole_image():
    adapter = clip_adapter()
    tensor = adapter.preprocess(Image.new('RGB', (40, 10)))
    assert tuple(tensor.shape) == (3, 8, 8) and adapter.preprocessing_geometry == 'resize'


# ---------------------------------------------------------------- registration and runner

def test_runner_is_registered():
    import check_partition_progress
    import experiment_common as common
    import download_datasets
    import xai_leaderboard
    config = yaml.safe_load((ROOT / 'src/configs/RefCOCO_epito.yaml').read_text())
    assert config['data']['name'] == 'RefCOCO' and config['data']['ground_truth']['format'] == 'refcoco'
    assert config['data']['mask_dir_final'] == 'RefCOCO_sam2_tiny_62_compact-sam_200'
    assert common.DEFAULT_CONFIGS['VL_RefCOCO'] == 'RefCOCO_epito'
    assert common.MODEL_NAMES['VL_RefCOCO'] == 'openai/clip-vit-base-patch32'
    assert common.EXPECTED_DATASETS['VL_RefCOCO'] == 'refcoco' and 'VL_RefCOCO' in common.LOCALIZATION_RUNNERS
    assert 'RefCOCO_epito' in check_partition_progress.DEFAULT_CONFIGS
    assert xai_leaderboard.DOMAINS['RefCOCO'] == 'vision-language'
    assert len(download_datasets.REFCOCO_ZIP_SHA256) == 64 and download_datasets.REFCOCO_ZIP.startswith('https://')
    files = download_datasets.MODELS['refcoco']
    assert {Path(path).name for _, path, _ in files} >= {'config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt'}
    assert all('/resolve/main/' not in url and len(digest) == 64 for url, _, digest in files)
    assert common.create_parser('VL_RefCOCO').parse_args([]).all_sentences is False
    import generate_refcoco_partitions
    assert generate_refcoco_partitions.create_parser(default_config='RefCOCO_epito').parse_args([]).config == 'RefCOCO_epito'


def test_download_rejects_other_splits(tmp_path):
    import download_datasets
    with pytest.raises(ValueError, match='val only'):
        download_datasets.main(['--dataset', 'refcoco', '--assets', 'dataset', '--split', 'test', '--root', str(tmp_path)])


def test_records_are_one_per_expression_and_share_the_image_partition(tmp_path):
    import experiment_common as common
    config, _ = dataset(tmp_path)
    inputs = common.prepare_inputs('VL_RefCOCO', common.create_parser('VL_RefCOCO').parse_args(['--config', str(config)]))
    ids = [record['image_id'] for record in inputs['records']]
    # --limit takes one object per image first: ref0 of image 1, ref2 of image 2, then ref1 of image 1.
    assert ids == [f'{STEMS[0]}__ref0_s0', f'{STEMS[1]}__ref2_s3', f'{STEMS[0]}__ref1_s2']
    assert {record['partition_id'] for record in inputs['records']} == set(STEMS)
    assert inputs['records'][0]['partition_path'].name == f'{STEMS[0]}_compact.npy'
    every = common.prepare_inputs('VL_RefCOCO', common.create_parser('VL_RefCOCO').parse_args(
        ['--config', str(config), '--all-sentences']))
    assert len(every['records']) == 4
    one = common.prepare_inputs('VL_RefCOCO', common.create_parser('VL_RefCOCO').parse_args(
        ['--config', str(config), '--image-ids', STEMS[1]]))
    assert [record['image_id'] for record in one['records']] == [f'{STEMS[1]}__ref2_s3']
    with pytest.raises(ValueError, match='Unknown RefCOCO'):
        common.prepare_inputs('VL_RefCOCO', common.create_parser('VL_RefCOCO').parse_args(
            ['--config', str(config), '--image-ids', 'nope']))


def test_explanation_run_credits_the_referred_object(tmp_path):
    import csv
    import experiment_common as common
    config, _ = dataset(tmp_path)
    args = common.create_parser('VL_RefCOCO').parse_args([
        '--config', str(config), '--show-gt', '--limit', '2', '--max-evals', '8', '--kernel-shap-samples', '10',
        '--auc-num-samples', '5', '--road-short-side', '0', '--no-heatmaps', '--verbose-level', 'low'])
    result = common.run('VL_RefCOCO', args, adapter=clip_adapter())
    assert result['completed'] == 2
    record = f'{STEMS[0]}__ref0_s0'
    folder = result['run_dir'] / record
    prediction = json.loads((folder / f'{record}_predictions.json').read_text())
    assert prediction['expression'] == 'red man right' and prediction['explained_class'] == 'red man right'
    assert prediction['ann_id'] == 11 and prediction['image_stem'] == STEMS[0]
    gt = json.loads((folder / f'{record}_gt.json').read_text())
    assert gt['annotation_ids'] == [11]
    values = np.load(folder / f'{record}_attributions.npz')['SAM_c'][0]
    assert values[:, 10:].mean() > values[:, :10].mean()
    table = list(csv.DictReader((result['evaluation_dir'] / 'auc_results_all_images_current_run.csv').open()))
    assert len(table) == 6
    for row in table:
        assert row['auc_ins'] and row['road_gap'] and row['best_iou'] and row['pointing_accuracy']
    info = json.loads((result['run_dir'] / 'run_info.json').read_text())
    assert info['score_type'] == 'image-text cosine similarity'


def test_road_sets_the_expression_before_scoring(tmp_path):
    import evaluate_road
    import experiment_common as common
    config, _ = dataset(tmp_path)
    args = common.create_parser('VL_RefCOCO').parse_args([
        '--config', str(config), '--limit', '1', '--max-evals', '8', '--kernel-shap-samples', '10',
        '--no-auc', '--no-road', '--no-plots', '--verbose-level', 'low'])
    common.run('VL_RefCOCO', args, adapter=clip_adapter())
    road_args = evaluate_road.parser(['--runner', 'VL_RefCOCO', '--config', str(config), '--limit', '1',
                                      '--max-evals', '8', '--road-short-side', '0']).parse_args(
        ['--runner', 'VL_RefCOCO', '--config', str(config), '--limit', '1', '--max-evals', '8', '--road-short-side', '0'])
    adapter = clip_adapter()
    summary = evaluate_road.run(road_args, adapter=adapter)
    assert adapter.expression == 'red man right' and summary


def test_summary_setup_counts_images_with_expressions(tmp_path):
    import summary_setup
    _, config = dataset(tmp_path)
    assert summary_setup.masks_found(config['data']) == set(STEMS)
    assert summary_setup.DATASETS['refcoco'].models == ('clip_b32',)


def test_clip_size_reads_dicts_sizedict_objects_and_ints():
    from types import SimpleNamespace
    from model_adapters import clip_size
    assert clip_size({'height': 224, 'width': 224}) == (224, 224)
    assert clip_size(SimpleNamespace(height=224, width=224, shortest_edge=None)) == (224, 224)
    assert clip_size(224) == (224, 224)
