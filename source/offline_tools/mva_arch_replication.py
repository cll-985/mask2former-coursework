"""Frozen two-seed Mask R-CNN controls and full-COCO max-single bootstrap."""
import csv
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.compare_fusion_rules import atomic_json_dump
from offline_tools.mva_fusion_controls import RULES, apply_rule
from offline_tools.mva_full_coco_bootstrap import Resampler, evaluate, full_ap, interval
from offline_tools.mva_score_floor_audit import metrics

GT = 'CarDD_COCO/annotations/instances_val2017.json'
LEFT = 'output_offline/R01_MVA_MaskRCNN_R50_20261017_gpu1_v1/seed42/inference/coco_instances_results.json'
RIGHT = 'output_offline/R01_MVA_MaskRCNN_seed3407_20261026_gpu0_resume_v3/stage/inference/coco_instances_results.json'
INPUT_HASHES = {
    GT: '25d445e2da4d86a14390d419d8147a1ace97f8f71f2e37cd2e7c6be79398a8d2',
    LEFT: '645edf384c65b30871a8bf1eeba4959aa7e4fe8ab7d3cf6328e83621fc4ec977',
    RIGHT: '12001a6dad31a60d0a6e6e4fa9e9f25dfca7f48faa520916503a96a7e2c6c396'}
EXPECTED_AP = (51.521617310192, 51.14292873772682)
FLOOR, RESAMPLES, SEED = .001, 2000, 20261016


def save(data, path):
    atomic_json_dump(data, str(path), indent=2)


def floor_output(values):
    return [x for x in values if float(x['score']) >= FLOOR]


def validate_predictions(dataset, values):
    images = {int(x['id']): x for x in dataset['images']}
    cats = {int(x['id']) for x in dataset['categories']}
    counts = Counter()
    for x in values:
        image = images.get(int(x['image_id']))
        if image is None or int(x['category_id']) not in cats:
            raise ValueError('Prediction outside frozen GT images/categories')
        score = float(x['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid prediction score')
        rle = x['segmentation']
        if rle['size'] != [image['height'], image['width']] or not isinstance(rle['counts'], str):
            raise ValueError('Invalid original-resolution RLE')
        counts[int(x['image_id'])] += 1
    if counts and max(counts.values()) > 100:
        raise ValueError('Source/output exceeds frozen total top100 per image')
    return {'detections': len(values), 'images_with_detections': len(counts),
            'empty_prediction_ids': sorted(set(images)-set(counts)),
            'below_uniform_floor': sum(float(x['score']) < FLOOR for x in values)}


def load_inputs():
    dataset = json.loads((ROOT/GT).read_text())
    if len(dataset['images']) != 810 or len(dataset['categories']) != 6:
        raise ValueError('Unexpected frozen dataset shape')
    fields = {k for image in dataset['images'] for k in
              ('vehicle_id', 'sequence_id', 'video_id', 'group_id') if k in image}
    if fields:
        raise ValueError('Group IDs found; image-independent resampling not authorized')
    values = [json.loads((ROOT/p).read_text()) for p in (LEFT, RIGHT)]
    return dataset, values, [validate_predictions(dataset,x) for x in values]


def max_single_delta(ap_left, ap_right, ap_candidate):
    return float(ap_candidate-max(ap_left,ap_right))


def sampled_delta(sampler, indices):
    dataset, predictions = sampler.draw(indices)
    aps = [full_ap(dataset,x) for x in predictions]
    return aps, max_single_delta(*aps)


def measured(dataset, predictions):
    evaluator = evaluate(dataset, predictions)
    result = metrics(evaluator)
    precision = evaluator.eval['precision']
    result['per_category_AP'] = {}
    for index, cid in enumerate(evaluator.params.catIds):
        p = precision[:,:,index,0,-1]; p = p[p > -1]
        result['per_category_AP'][str(cid)] = float(p.mean()*100) if p.size else None
    return result


def run(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CUDA must be hidden')
    dataset, raw, input_counts = load_inputs()
    singles = [floor_output(x) for x in raw]
    # Audit only GT category presence BEFORE any new candidate AP is evaluated.
    sampler = Resampler(dataset, singles)
    rng = np.random.RandomState(SEED)
    missing = sum(not sampler.complete_categories(rng.randint(0,len(sampler.ids),len(sampler.ids)))
                  for _ in range(RESAMPLES))
    protocol = {'image_count': 810, 'seed': SEED, 'resamples': RESAMPLES,
        'primary': 'mask_nms minus max(seed42 AP,seed3407 AP),max reselected every draw',
        'rules': list(RULES), 'mask_iou': .5, 'gaussian_sigma': .5,
        'score_floor_all_outputs': FLOOR, 'max_dets_total_per_image': 100,
        'group_metadata': 'no recognized group IDs; residual image dependence unknown',
        'category_missingness_draws': RESAMPLES,'category_missingness_count': missing,
        'test_split_access': False, 'gpu_disabled': True,
        'not_seed_population_inference': True,'partial_ci_not_stopping_criterion': True,
        'input_sha256': INPUT_HASHES,'input_counts': input_counts}
    save(protocol,output/'protocol.json')
    if missing > RESAMPLES*.01:
        raise RuntimeError('GT-only missingness >1%; freeze grouped protocol before analysis')
    report = {'status':'running','protocol':protocol,'singles':{},'rules':{}}
    for i,name in enumerate(('seed42','seed3407')):
        result = measured(dataset,singles[i])
        raw_ap = full_ap(dataset,raw[i]) if len(raw[i]) != len(singles[i]) else result['AP']
        if abs(raw_ap-EXPECTED_AP[i]) >= 1e-6:
            raise ValueError('Official single AP reproduction failed: '+name)
        report['singles'][name] = {'metrics':result,'raw_ap_reproduced':raw_ap,
                                 'counts':validate_predictions(dataset,singles[i])}
    reference = max(report['singles'][k]['metrics']['AP'] for k in report['singles'])
    controls = output/'controls'; controls.mkdir(exist_ok=False)
    primary = None
    for rule in RULES:
        tick = time.monotonic()
        predictions,stats = apply_rule(rule,raw[0],raw[1])
        before = len(predictions); predictions = floor_output(predictions)
        counts = validate_predictions(dataset,predictions)
        result = measured(dataset,predictions)
        path = controls/(rule+'_coco_instances_results.json')
        atomic_json_dump(predictions,str(path))
        report['rules'][rule] = {'metrics':result,'delta_vs_stronger_single':result['AP']-reference,
            'relative_gain_percent':(result['AP']/reference-1)*100,
            'counts':counts,'removed_by_uniform_output_floor':before-len(predictions),
            'rule_stats':stats,'seconds_including_eval':time.monotonic()-tick}
        save(report,output/'controls_report.json')
        print('RULE_COMPLETE {} AP={:.10f} delta={:.10f}'.format(rule,result['AP'],result['AP']-reference),flush=True)
        if rule == 'mask_nms':primary = predictions
    report['status'] = 'complete'; save(report,output/'controls_report.json')
    original = {'seed42_ap':report['singles']['seed42']['metrics']['AP'],
                'seed3407_ap':report['singles']['seed3407']['metrics']['AP'],
                'candidate_ap':report['rules']['mask_nms']['metrics']['AP']}
    original['delta_vs_stronger_single'] = max_single_delta(*[original[k] for k in ('seed42_ap','seed3407_ap','candidate_ap')])
    save(original,output/'primary_point_estimate.json')
    sampler = Resampler(dataset,singles+[primary])
    rng = np.random.RandomState(SEED); complete_deltas=[]; all_deltas=[]
    selected_left=selected_right=ties=0; started=time.monotonic()
    with (output/'resamples.csv').open('x',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['draw','seed42_ap','seed3407_ap','candidate_ap','max_single_ap','delta','complete_categories'])
        for index in range(RESAMPLES):
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 16*1024**2:
                raise RuntimeError('Own process high-water RSS above16GiB')
            indices=rng.randint(0,len(sampler.ids),len(sampler.ids))
            complete=sampler.complete_categories(indices)
            aps,delta=sampled_delta(sampler,indices)
            selected_left+=int(aps[0]>aps[1]);selected_right+=int(aps[1]>aps[0]);ties+=int(aps[0]==aps[1])
            all_deltas.append(delta)
            if complete:complete_deltas.append(delta)
            writer.writerow([index]+aps+[max(aps[:2]),delta,int(complete)])
            if (index+1)%10==0 or index==0 or index+1==RESAMPLES:
                handle.flush()
                progress={'completed':index+1,'target':RESAMPLES,'elapsed_seconds':time.monotonic()-started,
                    'complete_category_resamples':len(complete_deltas),'partial_ci_must_not_be_used_for_stopping':True}
                save(progress,output/'progress.json');print('BOOTSTRAP_PROGRESS '+json.dumps(progress),flush=True)
    final={'status':'complete','protocol':protocol,'original':original,
        'complete_category_conditional_interval':interval(complete_deltas),
        'unfiltered_sensitivity_interval':interval(all_deltas),
        'reference_reselected_every_draw':True,
        'reference_selection_counts':{'seed42':selected_left,'seed3407':selected_right,'ties':ties},
        'elapsed_seconds':time.monotonic()-started}
    save(final,output/'full_coco_bootstrap_report.json')
    print('ARCH_REPLICATION_COMPLETE',flush=True)
