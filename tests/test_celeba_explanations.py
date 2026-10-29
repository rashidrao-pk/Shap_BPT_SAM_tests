"""Notebook architecture, raw logits, attribute targets, and part-mask proxies."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np
from PIL import Image
import torch
import yaml

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src/scripts/explanations'),
               str(Path(__file__).resolve().parents[1] / 'src/scripts')]
from celeba_model import ATTRIBUTES, MEAN, STD, CelebModel, CelebaAdapter, attribute_names
from all_methods import lrp_model
from ground_truth import load_ground_truth
from download_celeba_checkpoint import download
import experiment_common as common


class CelebaExplanationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_strict_state_loading_preprocessing_and_raw_logits(self):
        model = CelebModel().eval()
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.copy_(torch.arange(40, dtype=torch.float32)-20)
        checkpoint = self.root / 'model.pth'
        torch.save(model.state_dict(), checkpoint)
        args = SimpleNamespace(checkpoint=checkpoint, model_batch_size=2,
                               class_names=None, attribute_file=None)
        adapter = CelebaAdapter(args, torch.device('cpu'))
        image = np.full((32, 48, 3), 255, dtype=np.uint8)
        values = adapter.predict_batch([image])[0]
        np.testing.assert_allclose(values, np.arange(40)-20)
        tensor = adapter.preprocess(Image.fromarray(image))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        np.testing.assert_allclose(tensor[:, 0, 0], (1-np.array(MEAN))/np.array(STD), rtol=1e-6)
        self.assertEqual(adapter.class_names[11], 'Brown_Hair')
        converted = lrp_model(adapter.model)
        with torch.no_grad():
            torch.testing.assert_close(converted(tensor[None]), adapter.model(tensor[None]))
        from captum.attr import LRP
        attr = LRP(converted).attribute(tensor[None].requires_grad_(True), target=11)
        self.assertTrue(torch.isfinite(attr).all())

    def test_attribute_header_and_proxy_mapping(self):
        header = self.root / 'attributes.txt'
        header.write_text('30000\nimage_id ' + ' '.join(ATTRIBUTES) + '\n')
        self.assertEqual(attribute_names(header), ATTRIBUTES)
        masks = self.root / 'masks/0'
        masks.mkdir(parents=True)
        hair = np.zeros((8, 10), dtype=np.uint8); hair[:2] = 255
        mouth = np.zeros_like(hair); mouth[5:7, 3:7] = 255
        Image.fromarray(hair).save(masks / '00000_hair.png')
        Image.fromarray(mouth).save(masks / '00000_mouth.png')
        config = {'data': {'ground_truth': {'format': 'celebamask_hq', 'root': str(masks.parent)}}}
        gt = load_ground_truth(config, '0.jpg', (8, 10, 3), predicted_class='Brown_Hair')
        self.assertEqual(gt['scope'], 'attribute_region_proxy')
        np.testing.assert_array_equal(gt['mask'], hair > 0)
        smile = load_ground_truth(config, '0.jpg', (8, 10, 3), predicted_class='Smiling')
        np.testing.assert_array_equal(smile['mask'], mouth > 0)
        unmapped = load_ground_truth(config, '0.jpg', (8, 10, 3), predicted_class='Male')
        self.assertEqual(unmapped['scope'], 'attribute_unmapped')
        self.assertFalse(unmapped['mask'].any())

    def test_default_target_is_brown_hair_not_highest_attribute(self):
        images = self.root / 'images'; images.mkdir()
        Image.fromarray(np.full((8, 10, 3), 100, dtype=np.uint8)).save(images / '0.jpg')
        partitions = self.root / 'partitions/face'; partitions.mkdir(parents=True)
        labels = np.indices((8, 10))[1] // 5
        np.save(partitions / '0_compact.npy', labels)
        (partitions / '0_results.json').write_text(json.dumps({'saved_partitions': {'compact': '0_compact.npy'}}))
        checkpoint = self.root / 'model.pth'; checkpoint.touch()
        config = self.root / 'config.yaml'
        config.write_text(yaml.safe_dump({'data': {'name': 'CelebAHQ', 'input_format': 'directory',
                     'image_dir': str(images), 'mask_dir_final': 'face'},
                     'device': 'cpu', 'model': {'checkpoint': str(checkpoint)},
                     'output': {'partitions': str(partitions.parent), 'explanations': str(self.root / 'x'),
                                'evaluations': str(self.root / 'e')}}))
        class Adapter:
            class_names = ATTRIBUTES
            def predict_batch(self, images):
                result = np.zeros((len(images), 40)); result[:, 11] = 3; result[:, 15] = 5
                return result
            def detections(self, image):
                return []
        args = common.create_parser('OC_CelebA').parse_args([
            '--config', str(config), '--max-evals', '8', '--no-auc', '--no-plots', '--no-html', '--torch-threads', '2'])
        result = common.run('OC_CelebA', args, adapter=Adapter())
        predictions = json.loads((result['run_dir'] / '0/0_predictions.json').read_text())
        self.assertEqual(predictions['explained_class'], 'Brown_Hair')
        self.assertEqual(predictions['top_k_classes'][0]['class_name'], 'Eyeglasses')
        self.assertIn('Brown_Hair', str(result['run_dir']))
        self.assertIn('Brown_Hair', str(result['evaluation_dir']))

    def test_bad_download_does_not_replace_checkpoint(self):
        destination = self.root / 'weights.pth'; destination.write_bytes(b'previous')
        def fake_download(url, output):
            Path(output).write_bytes(b'corrupted')
        with patch('download_celeba_checkpoint.urllib.request.urlretrieve', fake_download):
            with self.assertRaisesRegex(ValueError, 'checksum differs'):
                download(destination, overwrite=True)
        self.assertEqual(destination.read_bytes(), b'previous')
        self.assertFalse(destination.with_name(destination.name+'.part').exists())


if __name__ == '__main__':
    unittest.main()
