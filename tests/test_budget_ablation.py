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
            args=common.create_parser('OC_ResNet').parse_args(['--explain_all','--budget-selection',str(path)])
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
            folder=root/'eval/budget_ablation/OC_ResNet/fixture/bg_black/pilot'
            study=json.loads((folder/'study.json').read_text())
            rows=json.loads((folder/'comparisons.json').read_text())
            self.assertEqual(study['state'],'complete')
            self.assertEqual(len(rows),3*2*2*2)
            self.assertTrue((folder/'budget_report.html').is_file())
            self.assertTrue((folder/'reference_diagnostics.json').is_file())
            self.assertEqual(json.loads((folder/'selection.json').read_text())['status'],result['status'])
            restored=pilot.run(args,adapter=Adapter())
            self.assertEqual(restored['selected_budget'],result['selected_budget'])


if __name__=='__main__':
    unittest.main()
