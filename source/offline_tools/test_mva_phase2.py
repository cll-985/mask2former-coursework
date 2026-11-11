#!/usr/bin/env python3
"""Scientific correctness checks against direct official COCOeval."""
import copy
import json
import os
import sys
import unittest

import numpy as np
from pycocotools import mask as mask_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from offline_tools.mva_full_coco_bootstrap import Resampler, evaluate, full_ap, interval
from offline_tools.mva_fusion_controls import binary_geometry_control, soft_nms
from offline_tools.mva_fusion_mechanisms import maximum_matching, matching_summary
from offline_tools.mva_score_floor_audit import metrics as floor_metrics


def rle(values):
    result = mask_util.encode(np.asfortranarray(values.astype(np.uint8)))
    result['counts'] = result['counts'].decode('utf-8')
    return result


def fixture():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:5, 1:5] = 1
    seg = rle(mask)
    gt = {'info': {}, 'categories': [{'id': 1, 'name': 'damage'}],
          'images': [{'id': i, 'width': 8, 'height': 8} for i in (11, 29)],
          'annotations': [{'id': i, 'image_id': image_id, 'category_id': 1,
                           'segmentation': copy.deepcopy(seg), 'bbox': [1, 1, 4, 4],
                           'area': 16, 'iscrowd': 0} for i, image_id in enumerate((11, 29), 1)]}
    prediction = {'image_id': 11, 'category_id': 1, 'segmentation': seg,
                  'bbox': [1, 1, 4, 4], 'score': 0.9}
    return gt, prediction, mask


class Phase2Tests(unittest.TestCase):
    def test_score_floor_metric_extraction_matches_official_metric(self):
        gt, prediction, _ = fixture()
        extracted = floor_metrics(evaluate(gt, [prediction]))
        self.assertAlmostEqual(extracted['AP'], 100.0 * 51.0 / 101.0, places=9)
        self.assertAlmostEqual(extracted['AP50'], extracted['AP'], places=9)
        self.assertAlmostEqual(extracted['AP75'], extracted['AP'], places=9)

    def test_oracle_uses_augmenting_path_and_duplicate_fp_definition(self):
        self.assertEqual(maximum_matching([[0, 1], [0]], 2), 2)
        gt, prediction, _ = fixture()
        duplicate = copy.deepcopy(prediction)
        duplicate['score'] = 0.8
        result = matching_summary(evaluate(gt, [prediction, duplicate]), 0.5)
        self.assertEqual(result['tp'], 1)
        self.assertEqual(result['fp'], 1)
        self.assertEqual(result['duplicate_fp'], 1)

    def test_full_metric_is_not_mean_image_metric(self):
        gt, prediction, _ = fixture()
        ap = full_ap(gt, [prediction])
        self.assertAlmostEqual(ap, 100.0 * 51.0 / 101.0, places=9)
        evaluator = evaluate(gt, [prediction])
        # The public official summary must match our extracted full metric.
        import contextlib
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
            evaluator.summarize()
        self.assertAlmostEqual(ap, evaluator.stats[0] * 100.0, places=12)

    def test_duplicate_images_and_annotations_are_preserved(self):
        gt, prediction, _ = fixture()
        sampler = Resampler(gt, [[prediction]])
        duplicated, systems = sampler.draw([0, 0])
        self.assertEqual([x['id'] for x in duplicated['images']], [1, 2])
        self.assertEqual([x['id'] for x in duplicated['annotations']], [1, 2])
        self.assertEqual(len(systems[0]), 2)
        self.assertAlmostEqual(full_ap(duplicated, systems[0]), 100.0, places=9)
        duplicated, systems = sampler.draw([1, 1])
        self.assertAlmostEqual(full_ap(duplicated, systems[0]), 0.0, places=9)
        self.assertEqual([x['id'] for x in gt['images']], [11, 29])

    def test_identical_prediction_delta_and_interval_are_zero(self):
        gt, prediction, _ = fixture()
        sampler = Resampler(gt, [[prediction], [copy.deepcopy(prediction)]])
        deltas = []
        for draw in ([0, 0], [0, 1], [1, 1], [1, 0]):
            data, systems = sampler.draw(draw)
            deltas.append(full_ap(data, systems[1]) - full_ap(data, systems[0]))
        result = interval(deltas)
        self.assertEqual(result['ci95_low'], 0.0)
        self.assertEqual(result['ci95_high'], 0.0)

    def test_soft_nms_formula_and_input_integrity(self):
        _, a, _ = fixture()
        b = copy.deepcopy(a)
        b['score'] = 0.8
        before = json.dumps([a, b], sort_keys=True)
        linear, stats = soft_nms([a], [b], 'linear')
        self.assertEqual(len(linear), 1)
        self.assertEqual(stats['removed_below_floor'], 1)
        gaussian, _ = soft_nms([a], [b], 'gaussian')
        self.assertEqual(len(gaussian), 2)
        self.assertAlmostEqual(gaussian[1]['score'], 0.8 * np.exp(-2.0), places=12)
        self.assertEqual(json.dumps([a, b], sort_keys=True), before)

    def test_union_and_endpoint_degeneracies(self):
        _, a, left = fixture()
        right = np.zeros_like(left)
        right[1:5, 2:6] = 1
        b = copy.deepcopy(a)
        b['segmentation'] = rle(right)
        b['score'] = 0.8
        output, _ = binary_geometry_control([a], [b], 'endpoint')
        self.assertEqual(len(output), 1)
        self.assertTrue(np.array_equal(mask_util.decode(output[0]['segmentation']), left))
        output, _ = binary_geometry_control([a], [b], 'union')
        self.assertTrue(np.array_equal(mask_util.decode(output[0]['segmentation']), left | right))
        b['score'] = a['score']
        output, _ = binary_geometry_control([a], [b], 'endpoint')
        self.assertTrue(np.array_equal(mask_util.decode(output[0]['segmentation']), left | right))


if __name__ == '__main__':
    unittest.main(verbosity=2)
