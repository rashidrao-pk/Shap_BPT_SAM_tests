"""Budget stability, no hold-out peeking, shared scopes, and pilot execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/scripts/explanations'))
import budget_ablation as pilot
from budget_selection import DEFAULT_CRITERIA, compare_maps, select_budget
from all_methods import apply_budget_selection, method_names
import experiment_common as common


class BudgetAblationTest(unittest.TestCase):
    def study(self, runner='OC_ResNet'):
        study = {'state':'complete', 'runner':runner, 'dataset':'fixture', 'background':'black',
                 'budgets':[100,200,500], 'reference_budget':1000, 'methods':list(pilot.METHODS),
                 'repeats':2, 'compute_auc':True, 'auc_points':200, 'criteria':dict(DEFAULT_CRITERIA),
                 'splits':{'calibration':['a','b'], 'validation':['c']}, 'rows':[]}
        for image in ['a','b','c']:
            for method in pilot.METHODS:
                for budget in study['budgets']:
                    for repeat in range(2):
                        study['rows'].append({'image_id':image,'method':method,'budget':budget,'repeat':repeat,
                            'reference_stable':True,'informative_reference':True,
                            'spearman':.8 if budget==100 else 1.,'top_jaccard':1.,'relative_l1':0.,
                            'auc_ins_difference':0.,'auc_del_difference':0.})
        return study

    def test_smallest_stable_budget_then_independent_validation(self):
        study = self.study()
        result = select_budget([study])
        self.assertEqual(result['selected_budget'],200)
        self.assertEqual(result['status'],'validated')
        for row in study['rows']:
            if row['image_id']=='c' and row['budget']==200:
                row['spearman']=.8
        failed = select_budget([study])
        self.assertEqual(failed['status'],'validation_failed')
        self.assertEqual(failed['calibration_candidate'],200)
        self.assertIsNone(failed['selected_budget'])  # Never escalate to 500 using hold-out outcomes.

    def test_every_study_and_larger_budget_must_pass(self):
        first, second = self.study(), self.study('COD_ViT')
        for row in second['rows']:
            row['reference_stable']=False
        self.assertIsNone(select_budget([first,second])['selected_budget'])
        for row in first['rows']:
            if row['budget']==500:
                row['relative_l1']=.9
        self.assertIsNone(select_budget([first])['selected_budget'])
        first=self.study(); first['rows'].pop()
        with self.assertRaisesRegex(ValueError,'missing'):
            select_budget([first])

    def test_constant_reference_and_diagnostic_only_are_not_recommendations(self):
        result=compare_maps(np.zeros((2,2)),np.zeros((2,2)))
        self.assertFalse(result['informative_reference'])
        self.assertIsNone(result['spearman'])
        study=self.study();study['compute_auc']=False
        result=select_budget([study])
        self.assertEqual(result['status'],'stability_only')
        self.assertIsNone(result['selected_budget'])

    def test_single_budget_application_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'selection.json';path.write_text(json.dumps(select_budget([self.study()])))
            args=common.create_parser('OC_ResNet').parse_args(['--explain_all','--bg-type','black','--budget-selection',str(path)])
            apply_budget_selection(args,'OC_ResNet')
            self.assertEqual(args.max_evals,200)
            self.assertEqual(method_names(args)[:3],['BPT-200','BPT+SAM_c-200','SAM_c+KernelSHAP-200'])
            with self.assertRaisesRegex(ValueError,'not validated'):
                apply_budget_selection(args,'OD_DETR')
            path.write_text(json.dumps({'status':'validation_failed','selected_budget':None}))
            with self.assertRaisesRegex(ValueError,'did not validate'):
                apply_budget_selection(args,'OC_ResNet')

    def test_split_is_reproducible_and_disjoint(self):
        records=[{'image_id':str(i)} for i in range(20)]
        a,split=pilot.choose_examples(records,5,0,.4)
        self.assertEqual(split,pilot.choose_examples(records,5,0,.4)[1])
        self.assertEqual(len(split['calibration']),3)
        self.assertEqual(len(split['validation']),2)
        self.assertFalse(set(split['calibration']) & set(split['validation']))

    def test_end_to_end_pilot_and_resumable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);images=root/'images';images.mkdir()
            partitions=root/'partitions/fixture';partitions.mkdir(parents=True)
            for number in range(3):
                image=np.zeros((8,12,3),dtype=np.uint8)
                image[:,:,0]=np.arange(12)[None,:]*20
                Image.fromarray(image).save(images/f'image{number}.jpg')
                np.save(partitions/f'image{number}_compact.npy',np.indices((8,12))[1]//4)
                (partitions/f'image{number}_results.json').write_text(json.dumps({'saved_partitions':{'compact':'present'}}))
            config=root/'config.yaml';config.write_text(yaml.safe_dump({'device':'cpu',
                'data':{'name':'ImageNet','input_format':'directory','image_dir':str(images),'mask_dir_final':'fixture'},
                'output':{'partitions':str(partitions.parent),'explanations':str(root/'x'),'evaluations':str(root/'eval')}}))
            class Adapter:
                class_names=['object','other']
                def predict_batch(self,images):
                    scores=np.asarray([image[:,:,0].mean()/255 for image in images])
                    return np.column_stack([scores,1-scores])
            args=pilot.parser(['--config',str(config),'--budgets','4','8','--reference-budget','16',
                               '--repeats','2','--limit','3','--auc-num-samples','3','--torch-threads','2',
                               '--methods','BPT','BPT+SAM_c','--no-plots'])
            result=pilot.run(args,adapter=Adapter())
            folder=root/'eval/budget_ablation/OC_ResNet/fixture/bg_full/pilot'
            study=json.loads((folder/'study.json').read_text())
            rows=json.loads((folder/'comparisons.json').read_text())
            self.assertEqual(study['state'],'complete')
            self.assertEqual(len(rows),3*2*2*2)
            self.assertTrue((folder/'budget_report.html').is_file())
            self.assertTrue((folder/'conclusion.md').is_file())
            self.assertIn('<h2>Conclusion</h2>', (folder/'budget_report.html').read_text())
            self.assertTrue((folder/'reference_diagnostics.json').is_file())
            self.assertEqual(json.loads((folder/'selection.json').read_text())['status'],result['status'])
            restored=pilot.run(args,adapter=Adapter())
            self.assertEqual(restored['selected_budget'],result['selected_budget'])


if __name__=='__main__':
    unittest.main()


class CocoBudgetAblationTest(unittest.TestCase):
    def test_yolo_detr_and_sam_variants_share_reproducible_coco_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / 'val2017'
            images.mkdir()
            records = []
            for number in range(1, 7):
                image_id = f'{number:012d}'
                Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(images / f'{image_id}.jpg')
                records.append({'id': number, 'file_name': f'{image_id}.jpg', 'height': 8, 'width': 12})
            annotation = root / 'instances_val2017.json'
            annotation.write_text(json.dumps({'images': records, 'annotations': [], 'categories': []}))
            splits = []
            for model in ['sam1_vit_b', 'sam2_large']:
                folder_name = f'MSCOCO_{model}_62_compact-sam_200'
                folder = root / 'partitions' / folder_name
                folder.mkdir(parents=True)
                for record in records:
                    image_id = f"{record['id']:012d}"
                    np.save(folder / f'{image_id}_compact.npy', np.zeros((8, 12), dtype=np.uint16))
                    (folder / f'{image_id}_results.json').write_text(json.dumps({'saved_partitions': {'compact': 'present'}}))
                config = root / f'{model}.yaml'
                config.write_text(yaml.safe_dump({'device': 'cpu', 'data': {
                    'name': 'MS_COCO', 'image_dir': str(images), 'annotation_file': str(annotation),
                    'mask_dir_final': folder_name}, 'output': {
                    key: str(root / key) for key in ['partitions', 'explanations', 'evaluations']}}))
                for runner, detector in [('OD_yolo', 'yolo11s'), ('OD_DETR', 'facebook/detr-resnet-50')]:
                    args = pilot.parser(['--runner', runner, '--model', detector, '--config', str(config),
                        '--show-gt', '--limit', '5', '--bg-type', 'black', '--class-aggregation', 'sum',
                        '--budgets', '100', '200', '500', '1000', '2000', '--reference-budget', '4000',
                        '--repeats', '3', '--pilot-seed', '0', '--holdout-fraction', '0.4',
                        '--auc-num-samples', '200', '--study-name', 'mscoco_pilot', '--dry-run'])
                    study = pilot.run(args)
                    self.assertEqual(study['runner'], runner)
                    self.assertEqual(study['dataset'], 'MS_COCO')
                    self.assertEqual(study['eligible_images'], 6)
                    self.assertEqual(len(study['splits']['calibration']), 3)
                    self.assertEqual(len(study['splits']['validation']), 2)
                    self.assertEqual(study['methods'], pilot.METHODS)
                    splits.append(study['splits'])
            self.assertTrue(all(split == splits[0] for split in splits))


class BudgetConclusionTest(unittest.TestCase):
    def test_conclusion_recommends_only_the_validated_budget(self):
        study = BudgetAblationTest().study('OD_yolo')
        study['dataset'] = 'MS_COCO'
        selection = select_budget([study])
        summary, text = pilot.budget_conclusion(selection, [study], Path('combined/selection.json'))
        self.assertIn('Recommended budget: 200', summary)
        self.assertIn('smallest tested common budget', summary)
        self.assertIn('MS_COCO', text)
        self.assertIn('--budget-selection combined/selection.json', text)
        self.assertIn('untested models', text)

    def test_failed_and_diagnostic_studies_never_recommend_a_budget(self):
        study = BudgetAblationTest().study('OD_DETR')
        for row in study['rows']:
            if row['image_id'] == 'c' and row['budget'] == 200:
                row['spearman'] = 0.1
        summary, text = pilot.budget_conclusion(select_budget([study]), [study], Path('selection.json'))
        self.assertIn('No budget is recommended', summary)
        self.assertIn('failed held-out validation', summary)
        self.assertIn('Failed held-out groups', text)
        self.assertNotIn('## Use this budget', text)
        study = BudgetAblationTest().study()
        study['compute_auc'] = False
        summary, text = pilot.budget_conclusion(select_budget([study]), [study], Path('selection.json'))
        self.assertIn('No production budget is recommended', summary)
        self.assertIn('AUC was disabled', summary)
        study = BudgetAblationTest().study()
        for row in study['rows']:
            row['reference_stable'] = False
        summary, text = pilot.budget_conclusion(select_budget([study]), [study], Path('selection.json'))
        self.assertIn('No tested common budget is recommended', summary)
        self.assertIn('unstable reference', text)

    def test_partial_method_study_cannot_claim_all_three_methods(self):
        study = BudgetAblationTest().study()
        study['methods'] = ['BPT']
        study['rows'] = [row for row in study['rows'] if row['method'] == 'BPT']
        summary, text = pilot.budget_conclusion(select_budget([study]), [study], Path('selection.json'))
        self.assertIn('omitted methods', summary)
        self.assertNotIn('## Use this budget', text)
