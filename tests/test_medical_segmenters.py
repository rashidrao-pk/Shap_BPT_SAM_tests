"""MedSAM box-grid and MedicoSAM priors (TO_DO T2).

The pipeline tests use a small fake MedSAM (same encoder/prompt/decoder interface), so they run on CPU
without weights. Tests with the real models skip, with the reason shown by `pytest -rs`, when the
checkpoints are absent; set MEDSAM_CHECKPOINT / MEDICOSAM_CHECKPOINT or use the default paths in
checkpoints/medsam/ and checkpoints/medicosam/.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
torch = pytest.importorskip('torch')
import medical_segmenters as med  # noqa: E402
import generate_coco_partitions as batch  # noqa: E402
from check_partition_progress import sam_label  # noqa: E402

SAM_KEYS = {'segmentation', 'area', 'bbox', 'predicted_iou', 'stability_score', 'point_coords', 'crop_box'}


def disk_image(size=256, radius=40):
    yy, xx = np.mgrid[:size, :size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius ** 2
    image = np.full((size, size, 3), 30, dtype=np.uint8)
    image[disk] = 220
    return image, disk


class FakeMedSAM(torch.nn.Module):
    """Decoder output for a box: the bright pixels inside it (from the encoder's input), as +-10 logits."""

    def __init__(self):
        super().__init__()
        self.prompt_encoder = self
        self.mask_decoder = self._decode
        self.encoded = None

    def image_encoder(self, tensor):
        self.encoded = tensor
        return torch.zeros(1, 256, 64, 64)

    def get_dense_pe(self):
        return torch.zeros(1, 256, 64, 64)

    def __call__(self, points=None, boxes=None, masks=None):  # prompt encoder
        return boxes, None

    def _decode(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        assert multimask_output is False
        bright = torch.nn.functional.interpolate(self.encoded[:, :1], size=(256, 256), mode='area')[0, 0] > .5
        logits, ious = [], []
        for x0, y0, x1, y1 in (sparse_prompt_embeddings[:, 0] / 4).tolist():
            inside = torch.zeros(256, 256, dtype=torch.bool)
            inside[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
            mask = inside & bright
            logits.append(torch.where(mask, 10., -10.)[None])
            ious.append(.9 if mask.any() else .1)
        return torch.stack(logits), torch.tensor(ious)[:, None]


def test_box_grid_covers_three_scales_with_half_side_stride():
    boxes = med.box_grid(256, 256)
    sides = np.unique(boxes[:, 2] - boxes[:, 0])
    assert sides.tolist() == [32, 64, 128]
    assert len(boxes) == 9 + 49 + 225
    assert boxes[:, [0, 1]].min() == 0 and boxes[:, 2].max() == 256 and boxes[:, 3].max() == 256
    wide = med.box_grid(100, 300)  # Edge row/column flush with the image border.
    assert wide[:, 2].max() == 300 and wide[:, 3].max() == 100


def test_preprocessing_follows_medsam_inference():
    from skimage import transform
    image, _ = disk_image(200)
    grey = image[..., 0]
    reference = transform.resize(np.repeat(grey[:, :, None], 3, axis=-1), (1024, 1024), order=3,
                                 preserve_range=True, anti_aliasing=True).astype(np.uint8)
    reference = (reference - reference.min()) / np.clip(reference.max() - reference.min(), 1e-8, None)
    result = med.medsam_preprocess(grey)
    assert result.shape == (1024, 1024, 3) and result.dtype == np.float32
    np.testing.assert_allclose(result, reference, atol=1e-6)
    np.testing.assert_allclose(med.medsam_preprocess(image), result, atol=1e-6)  # Grey as RGB: same input.


def test_box_grid_generator_returns_sam_annotations_covering_the_disk():
    image, disk = disk_image()
    generator = med.MedSAMBoxGridGenerator(FakeMedSAM(), 'cpu', points_per_batch=50)
    annotations = generator.generate(image)
    assert annotations and all(SAM_KEYS <= set(item) for item in annotations)
    item = annotations[0]
    assert item['segmentation'].dtype == bool and item['segmentation'].shape == disk.shape
    assert item['area'] == int(item['segmentation'].sum()) and len(item['bbox']) == 4
    best = max((a['segmentation'] & disk).sum() / (a['segmentation'] | disk).sum() for a in annotations)
    assert best > .5
    # Box NMS at 0.7: no two kept masks have near-identical boxes.
    boxes = torch.tensor([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (a['bbox'] for a in annotations)],
                         dtype=torch.float32)
    from torchvision.ops import box_iou
    overlap = box_iou(boxes, boxes).fill_diagonal_(0)
    assert overlap.max() <= .7 + 1e-6


def test_relaxing_thresholds_reuses_the_decoder_outputs():
    image, _ = disk_image()
    model = FakeMedSAM()
    calls = []
    original = model.image_encoder
    model.image_encoder = lambda tensor: (calls.append(1), original(tensor))[1]
    generator = med.MedSAMBoxGridGenerator(model, 'cpu', pred_iou_thresh=.95)
    assert generator.generate(image) == []  # Every fake IoU is 0.9 or 0.1.
    generator.pred_iou_thresh = .5
    assert generator.generate(image)
    assert len(calls) == 1


def test_state_dicts_are_found_in_training_checkpoints(tmp_path):
    weights = {'image_encoder.w': torch.ones(2), 'mask_decoder.b': torch.zeros(1)}
    for name, content in [('plain', weights), ('nested', {'model_state': {f'sam.{k}': v for k, v in weights.items()}}),
                          ('module', {'state_dict': {f'module.{k}': v for k, v in weights.items()}})]:
        torch.save(content, tmp_path / f'{name}.pt')
        state, info = med.read_state_dict(tmp_path / f'{name}.pt')
        assert set(state) == set(weights), name
    assert len(med.sha256(tmp_path / 'plain.pt')) == 64


def test_folder_names_labels_and_resume_settings():
    args = argparse.Namespace(model='medsam', medsam_prompt='box-grid', checkpoint=None, max_regions=254,
                              generate_at_least_n_masks=50, save_partitions=['sam', 'compact'],
                              compact_strategy='greedy', mso_alpha=0.5)
    assert batch.partition_folder_name({'name': 'BUSI'}, args) == 'BUSI_medsam_vitb_254_compact-sam_50'
    args.medsam_prompt = 'point-grid'
    assert batch.partition_folder_name({'name': 'BUSI'}, args).startswith('BUSI_medsam_vitb-pointgrid_254')
    args.model = 'medicosam'
    assert batch.partition_folder_name({'name': 'BUSI'}, args) == 'BUSI_medicosam_vitb_254_compact-sam_50'
    assert sam_label('BUSI_medsam_vitb_254_compact-sam_50') == 'MedSAM-VB'
    assert sam_label('BUSI_medsam_vitb-pointgrid_254_compact-sam_50') == 'MedSAM-VB-PG'
    assert sam_label('BUSI_medicosam_vitb_62_compact-sam_50') == 'MedicoSAM-VB-K62'
    parsed = batch.create_parser().parse_args(['--model', 'sam2'])
    assert 'medsam_prompt' not in batch.generation_settings(parsed)  # SAM-2 resume stamps unchanged.
    parsed = batch.create_parser().parse_args(['--model', 'medsam'])
    assert batch.generation_settings(parsed)['medsam_prompt'] == 'box-grid'


def checkpoint(name):
    path = Path(os.environ.get(f'{name.upper()}_CHECKPOINT', med.DEFAULT_CHECKPOINTS[name]))
    if not path.is_file():
        pytest.skip(f'{name} checkpoint not found at {path}')
    return path


@pytest.mark.parametrize('name', ['medsam', 'medicosam'])
def test_real_models_segment_a_bright_disk(name):
    path = checkpoint(name)
    pytest.importorskip('segment_anything')
    import generate_sam_partitions as sam
    args = batch.create_parser().parse_args(['--model', name, '--checkpoint', str(path)])
    generator, _ = med.build_medical_generator(args, 'cuda' if torch.cuda.is_available() else 'cpu',
                                               sam.sam1_generator_kwargs(args))
    image, disk = disk_image()
    annotations = generator.generate(image)
    assert annotations and SAM_KEYS <= set(annotations[0])
    best = max((a['segmentation'] & disk).sum() / (a['segmentation'] | disk).sum() for a in annotations)
    assert best > .5
    assert len(generator.segmenter_info['checkpoint_sha256']) == 64
