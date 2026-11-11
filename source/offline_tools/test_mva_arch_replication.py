"""Native CPU scientific regression for max-single and uniform-floor semantics."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from offline_tools.mva_arch_replication import (
    FLOOR,RESAMPLES,SEED,floor_output,max_single_delta,sampled_delta,validate_predictions)
from offline_tools.mva_full_coco_bootstrap import Resampler,full_ap
from offline_tools.test_mva_phase2 import fixture
from offline_tools.mva_fusion_controls import RULES,apply_rule


class Replication(unittest.TestCase):
    def test_frozen_constants(self):
        self.assertEqual((FLOOR,RESAMPLES,SEED),(.001,2000,20261016))
        self.assertEqual(len(RULES),8)

    def test_max_reference_changes_with_draw(self):
        self.assertEqual(max_single_delta(70,60,75),5)
        self.assertEqual(max_single_delta(40,80,75),-5)

    def test_actual_full_coco_reselects_reference_and_keeps_duplicate_images(self):
        gt,left,_=fixture();right=copy.deepcopy(left);right['image_id']=29
        sampler=Resampler(gt,[[left],[right],[left,right]])
        aps,d=sampled_delta(sampler,[0,0]);self.assertAlmostEqual(d,0,places=9)
        self.assertGreater(aps[0],aps[1])
        aps,d=sampled_delta(sampler,[1,1]);self.assertAlmostEqual(d,0,places=9)
        self.assertGreater(aps[1],aps[0])
        aps,d=sampled_delta(sampler,[0,1]);self.assertGreater(d,0)

    def test_same_predictions_zero_delta(self):
        gt,p,_=fixture();sampler=Resampler(gt,[[p],[p],[p]])
        for draw in ([0,0],[0,1],[1,1]):
            _,d=sampled_delta(sampler,draw);self.assertEqual(d,0)

    def test_floor_inclusive_and_inputs_not_mutated(self):
        values=[{'score':.0009},{'score':.001},{'score':.8}]
        before=copy.deepcopy(values)
        self.assertEqual([x['score'] for x in floor_output(values)],[.001,.8])
        self.assertEqual(values,before)

    def test_all_eight_rules_share_output_floor(self):
        gt,p,_=fixture();p['score']=.0015
        for rule in RULES:
            result,_=apply_rule(rule,[p],[])
            filtered=floor_output(result)
            self.assertTrue(all(x['score']>=.001 for x in filtered))
            validate_predictions(gt,filtered)
        result,_=apply_rule('consensus',[p],[])
        self.assertEqual(floor_output(result),[])

    def test_empty_predictions_keep_all_gt_images(self):
        gt,p,_=fixture()
        self.assertEqual(validate_predictions(gt,[p])['empty_prediction_ids'],[29])
        self.assertAlmostEqual(full_ap(gt,[p]),100*51/101,places=9)

    def test_invalid_ids_scores_size_and_budget_fail(self):
        gt,p,_=fixture()
        for change in ({'image_id':1000},{'score':float('nan')},{'category_id':99}):
            q=copy.deepcopy(p);q.update(change)
            with self.assertRaises(ValueError):validate_predictions(gt,[q])
        q=copy.deepcopy(p);q['segmentation']['size']=[9,9]
        with self.assertRaises(ValueError):validate_predictions(gt,[q])
        with self.assertRaises(ValueError):validate_predictions(gt,[p]*101)


if __name__=='__main__':unittest.main(verbosity=2)
