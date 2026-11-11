"""GT-only single-union-edit oracle diagnostic; NOT a deployable method or AP bound."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.mva_geometry_probe import ENV, GT, EXPECTED_GT, BASES, RCNN, sha, save

SPEC = '.aris/compute/mva-single-edit-probe-20261102.md'
SOURCES = ['offline_tools/mva_single_edit_probe.py', 'offline_tools/mva_geometry_probe.py',
           'offline_tools/mva_fusion_controls.py', 'offline_tools/compare_fusion_rules.py',
           'offline_tools/fuse_coco_instances.py', 'offline_tools/mva_full_coco_bootstrap.py', SPEC,
           '.aris/compute/mva-assignment-probe-20261102.md',
           '.aris/compute/mva-geometry-probe-20261101.md', '.aris/compute/mva-arch-analysis-20261030.md']
INPUT_HASHES = {
    GT: EXPECTED_GT,
    BASES['42']: 'bee0182e150ca5a02132993ffb7ced95100475ce5a9b355ab6a613e8d435b7a0',
    BASES['3407']: '33c6eb1a73dfdfea7d79f1d13778b1b7dd2f66d9487989d72b6b4e9b0471577c',
    BASES['2026']: 'd9f91dfed7667afb07f8e9010dc74910745aeee9b44940e81ba4cb1de293f65d',
    RCNN[0]: '645edf384c65b30871a8bf1eeba4959aa7e4fe8ab7d3cf6328e83621fc4ec977',
    RCNN[1]: '12001a6dad31a60d0a6e6e4fa9e9f25dfca7f48faa520916503a96a7e2c6c396'}
POSITIVE = 1e-12
ORACLES = ('delta_max_IoU', 'isolated_crossing_gain', 'own_actual_TP_gain', 'full_delta_U')


def isolated_evaluate(dataset, predictions):
    """COCO mutates annotation objects; each evaluator owns independent inputs."""
    from offline_tools.mva_full_coco_bootstrap import evaluate
    return evaluate(copy.deepcopy(dataset), copy.deepcopy(predictions))


def all_records(evaluator):
    return {(x['image_id'], x['category_id']): x for x in evaluator.evalImgs
            if x is not None and x['aRng'] == evaluator.params.areaRng[0]}


def tp(record):
    return (record['dtMatches'] > 0) & ~record['dtIgnore'].astype(bool)


def image_weights(predictions, thresholds):
    """Fixed image-global order, including cross-class ranks; not COCO AP weights."""
    import numpy as np
    by_image = defaultdict(list)
    for dtid, item in enumerate(predictions, 1):
        by_image[int(item['image_id'])].append(dtid)
    weights = {}
    for ids in by_image.values():
        n = len(ids)
        scores = [(float(predictions[i-1]['score']), int(predictions[i-1]['category_id'])) for i in ids]
        assert scores == sorted(scores, key=lambda x: (-x[0], x[1])), 'Input rank changed'
        w = np.cumsum((1.0/np.arange(1, n+1))[::-1])[::-1]/(thresholds*n)
        weights.update({dtid: float(value) for dtid, value in zip(ids, w)})
    return weights, dict(by_image)


def utilities(evaluator, weights, image_ids):
    import numpy as np
    result = {int(i): 0.0 for i in image_ids}
    for key, record in all_records(evaluator).items():
        w = np.asarray([weights[i] for i in record['dtIds']])
        result[key[0]] += float((tp(record)*w).sum())
    return result


def replay_one(endpoint, union, key, dtid, before, after):
    """Use official evaluateImg with precisely one IoU row and DT area replaced."""
    import numpy as np
    assert before['dtIds'] == after['dtIds'] and before['dtScores'] == after['dtScores']
    row = before['dtIds'].index(dtid)
    old_ious = endpoint.ious[key]
    new_ious = np.array(old_ious, copy=True)
    if new_ious.size:
        new_ious[row] = np.asarray(union.ious[key])[row]
    old_dt = next(x for x in endpoint._dts[key] if x['id'] == dtid)
    union_dt = next(x for x in union._dts[key] if x['id'] == dtid)
    saved = {k: old_dt[k] for k in ('area', 'segmentation')}
    saved_gt_ignore = [(g, '_ignore' in g, g.get('_ignore')) for g in endpoint._gts[key]]
    try:
        endpoint.ious[key] = new_ious if new_ious.size else old_ious
        old_dt.update({k: union_dt[k] for k in saved})
        result = endpoint.evaluateImg(key[0], key[1], endpoint.params.areaRng[0], 100)
    finally:
        endpoint.ious[key] = old_ious
        old_dt.update(saved)
        for g, present, value in saved_gt_ignore:
            if present: g['_ignore'] = value
            else: g.pop('_ignore', None)
    assert result['dtIds'] == before['dtIds'] and result['dtScores'] == before['dtScores']
    assert np.array_equal(result['dtIgnore'], before['dtIgnore']), 'All-area DT ignore changed'
    return result


def edit_label(endpoint, union, key, dtid, before, after, weights):
    import numpy as np
    single = replay_one(endpoint, union, key, dtid, before, after)
    diff = tp(single).astype(int)-tp(before).astype(int)
    row = before['dtIds'].index(dtid)
    w = np.asarray([weights[i] for i in before['dtIds']])
    full = float((diff*w).sum())
    own = float(diff[:, row].sum()*weights[dtid])
    eligible = [i for i, g in enumerate(endpoint._gts[key]) if not g.get('ignore', 0)]
    a, b = np.asarray(endpoint.ious[key]), np.asarray(union.ious[key])
    ja = float(a[row, eligible].max()) if a.size and eligible else 0.0
    jb = float(b[row, eligible].max()) if b.size and eligible else 0.0
    crossing = int((jb >= endpoint.params.iouThrs).sum()-(ja >= endpoint.params.iouThrs).sum())
    return dict(image_id=key[0], category_id=key[1], dtid=dtid,
                score=float(before['dtScores'][row]), utility_weight=weights[dtid], endpoint_max_IoU=ja,
                union_max_IoU=jb, delta_max_IoU=jb-ja,
                isolated_crossing_gain=float(crossing*weights[dtid]),
                own_actual_TP_gain=own, weighted_neighbor_externality=full-own,
                full_delta_U=full, neighbor_tp_state_changes=int(np.count_nonzero(np.delete(diff, row, axis=1))),
                own_tp_delta_by_threshold=json.dumps(diff[:, row].tolist()),
                weighted_full_delta_by_threshold=json.dumps((diff*w).sum(axis=1).tolist())), single


def choose(rows, image_ids):
    """Positive threshold and stable first candidate ties are frozen in advance."""
    selected = {name: {int(i): None for i in image_ids} for name in ORACLES}
    best = {name: {int(i): POSITIVE for i in image_ids} for name in ORACLES}
    for row in rows:
        for name in ORACLES:
            image = row['image_id']
            if row[name] > best[name][image]:
                best[name][image], selected[name][image] = row[name], row
    return selected


def ap(evaluator):
    p = evaluator.eval['precision'][:, :, :, 0, -1]
    return float(p[p > -1].mean()*100)


def self_test():
    import numpy as np
    from pycocotools import mask as mu
    def mask(indices):
        m = np.zeros((8, 8), dtype=np.uint8)
        for i in indices: m[i//8, i % 8] = 1
        return m
    def rle(m):
        r = mu.encode(np.asfortranarray(m)); r['counts'] = r['counts'].decode('ascii'); return r
    def fixture(gts, before_masks, after_masks):
        dataset = dict(images=[dict(id=1, height=8, width=8)], categories=[dict(id=1, name='x')],
            annotations=[dict(id=i, image_id=1, category_id=1, segmentation=rle(g),
                              area=int(g.sum()), bbox=mu.toBbox(rle(g)).tolist(), iscrowd=0)
                         for i, g in enumerate(gts, 1)])
        def predictions(masks):
            return [dict(image_id=1, category_id=1, score=.9-.1*i, segmentation=rle(m)) for i, m in enumerate(masks)]
        pred, other = predictions(before_masks), predictions(after_masks)
        saved_dataset, saved_pred, saved_other = copy.deepcopy((dataset, pred, other))
        a, b = isolated_evaluate(dataset, pred), isolated_evaluate(dataset, other)
        assert dataset == saved_dataset and pred == saved_pred and other == saved_other
        # Different official evaluators must not share the mutable GT objects.
        for original, ga, gb in zip(dataset['annotations'], a._gts[(1, 1)], b._gts[(1, 1)]):
            assert ga is not original and gb is not original and ga is not gb
        before, after = all_records(a)[(1, 1)], all_records(b)[(1, 1)]
        weights, _ = image_weights(pred, len(a.params.iouThrs))
        checked = []
        for dtid in range(1, len(pred)+1):
            old_matrix = np.array(a.ious[(1, 1)], copy=True)
            old_fields = [(float(x['area']), copy.deepcopy(x['segmentation'])) for x in a._dts[(1, 1)]]
            old_gt_ignore = [('_ignore' in g, g.get('_ignore')) for g in a._gts[(1, 1)]]
            label, cached = edit_label(a, b, (1, 1), dtid, before, after, weights)
            modified = copy.deepcopy(pred); modified[dtid-1] = copy.deepcopy(other[dtid-1])
            actual = isolated_evaluate(dataset, modified); full = all_records(actual)[(1, 1)]
            assert np.array_equal(cached['dtMatches'], full['dtMatches'])
            assert np.array_equal(cached['dtIgnore'], full['dtIgnore'])
            delta = utilities(actual, weights, [1])[1]-utilities(a, weights, [1])[1]
            assert abs(delta-label['full_delta_U']) < 1e-12
            assert abs(label['full_delta_U']-label['own_actual_TP_gain']-label['weighted_neighbor_externality']) < 1e-12
            assert np.array_equal(old_matrix, a.ious[(1, 1)])
            assert old_fields == [(float(x['area']), x['segmentation']) for x in a._dts[(1, 1)]]
            assert old_gt_ignore == [('_ignore' in g, g.get('_ignore')) for g in a._gts[(1, 1)]]
            # Replaying the restored official baseline must reproduce all matches.
            restored = replay_one(a, a, (1, 1), dtid, before, before)
            assert np.array_equal(restored['dtMatches'], before['dtMatches'])
            assert old_gt_ignore == [('_ignore' in g, g.get('_ignore')) for g in a._gts[(1, 1)]]
            assert dataset == saved_dataset and pred == saved_pred and other == saved_other
            checked.append(label)
        return checked
    g = mask(range(8)); expanded = mask(range(16))
    loss = fixture([g], [g], [expanded])
    assert loss[0]['full_delta_U'] < 0 and loss[0]['weighted_neighbor_externality'] == 0
    duplicate = fixture([g], [g, g], [expanded, g])
    assert duplicate[0]['neighbor_tp_state_changes'] > 0 and duplicate[0]['weighted_neighbor_externality'] > 0
    g1, g2 = mask(range(3)), mask(range(3, 9))
    competition = fixture([g1, g2], [mask(range(6)), g2], [mask(range(9)), g2])
    assert competition[0]['delta_max_IoU'] > 0 and competition[0]['neighbor_tp_state_changes'] > 0
    no_gt = fixture([], [g], [expanded])
    assert no_gt[0]['full_delta_U'] == no_gt[0]['delta_max_IoU'] == 0
    empty = isolated_evaluate(dict(images=[dict(id=1, height=8, width=8)], categories=[dict(id=1, name='x')], annotations=[]), [])
    assert utilities(empty, {}, [1]) == {1: 0.0}
    selected = choose([dict(image_id=1, dtid=1, **{k: 0.0 for k in ORACLES})], [1, 2])
    assert all(all(v is None for v in policy.values()) for policy in selected.values())
    tied = [dict(image_id=1, dtid=i, **{k: 1.0 for k in ORACLES}) for i in (1, 2)]
    assert all(policy[1]['dtid'] == 1 for policy in choose(tied, [1]).values())
    print(json.dumps(dict(status='passed', spec_revision=2, native_fixtures=5, cached_full_single_edits=6,
                          additional_selection_checks=2, positive_threshold=POSITIVE)), flush=True)


def probe(folder):
    from offline_tools.mva_fusion_controls import binary_geometry_control
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for path, expected in INPUT_HASHES.items():
        assert sha(ROOT/path) == expected, 'Frozen input changed: '+path
    dataset = json.loads((ROOT/GT).read_text()); image_ids = sorted(int(x['id']) for x in dataset['images'])
    pairs = [('mask2former_'+a+'_'+b, [BASES[a], BASES[b]]) for a, b in [('42', '3407'), ('42', '2026'), ('3407', '2026')]]
    pairs.append(('maskrcnn_42_3407', RCNN))
    report = dict(status='running', spec_revision=2, exploratory=True, gpu_disabled=True, test_split_access=False,
        scope='GT-only oracle, one union edit maximum per image; proxy-U oracle is NOT AP upper bound or deployable performance',
        utility='image-global fixed-rank average prefix precision across 10 IoU thresholds; not class-macro COCO AP',
        menu='all endpoint/union RLE representation differences; fixed .001 floor/top100/scores/order; exact-score ties union in both controls',
        positive_threshold=POSITIVE, image_count=len(image_ids), input_sha256=INPUT_HASHES, pairs={})
    for label, paths in pairs:
        raw = [json.loads((ROOT/p).read_text()) for p in paths]
        outputs = [binary_geometry_control(raw[0], raw[1], kind)[0] for kind in ('endpoint', 'union')]
        outputs = [[x for x in values if float(x['score']) >= .001] for values in outputs]
        assert len(outputs[0]) == len(outputs[1])
        changed = []
        for dtid, (a, b) in enumerate(zip(*outputs), 1):
            assert all(a[k] == b[k] for k in ('image_id', 'category_id', 'score'))
            if a['segmentation'] != b['segmentation']: changed.append(dtid)
        endpoint, union = [isolated_evaluate(dataset, values) for values in outputs]
        base_records, union_records = all_records(endpoint), all_records(union)
        weights, by_image = image_weights(outputs[0], len(endpoint.params.iouThrs))
        baseline_u = utilities(endpoint, weights, image_ids)
        rows = []
        fields = None
        with (folder/(label+'_single_edits.csv')).open('x', newline='') as f:
            writer = None
            for count, dtid in enumerate(changed, 1):
                item = outputs[0][dtid-1]; key = (int(item['image_id']), int(item['category_id']))
                row, _ = edit_label(endpoint, union, key, dtid, base_records[key], union_records[key], weights)
                if writer is None:
                    fields = list(row); writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
                writer.writerow(row); rows.append(row)
                if count % 500 == 0:
                    f.flush(); print('SINGLE_EDIT_PROGRESS {} {}/{}'.format(label, count, len(changed)), flush=True)
        selected = choose(rows, image_ids); result = dict(candidate_count=len(outputs[0]),
            rle_representation_difference_count=len(changed), label_rows=len(rows), endpoint_AP=ap(endpoint),
            endpoint_mean_U=sum(baseline_u.values())/len(image_ids),
            single_edits_positive_full_U=sum(x['full_delta_U'] > POSITIVE for x in rows),
            positive_full_U_with_nonpositive_own_actual_TP=sum(x['full_delta_U'] > POSITIVE and x['own_actual_TP_gain'] <= POSITIVE for x in rows),
            policies={}, selection_differences_vs_full_U={})
        choice_records = []
        for name, policy in selected.items():
            replacements = {row['dtid']: row for row in policy.values() if row is not None}
            predictions = [copy.deepcopy(outputs[1][i-1] if i in replacements else x) for i, x in enumerate(outputs[0], 1)]
            actual = isolated_evaluate(dataset, predictions); actual_u = utilities(actual, weights, image_ids)
            for image in image_ids:
                row = policy[image]; expected = baseline_u[image]+(row['full_delta_U'] if row is not None else 0.0)
                assert abs(actual_u[image]-expected) < 1e-10, 'Selected single-edit/full replay diverged'
                choice_records.append(dict(policy=name, image_id=image, selected_dtid=row['dtid'] if row else 0,
                    mean_proxy_gain=row['full_delta_U'] if row else 0.0))
            result['policies'][name] = dict(edited_images=len(replacements), AP=ap(actual), delta_AP=ap(actual)-ap(endpoint),
                mean_U=sum(actual_u.values())/len(image_ids),
                mean_delta_U=sum(actual_u[i]-baseline_u[i] for i in image_ids)/len(image_ids),
                harmful_proxy_images=sum(actual_u[i] < baseline_u[i]-POSITIVE for i in image_ids))
            result['selection_differences_vs_full_U'][name] = sum(
                (row['dtid'] if row else 0) != (selected['full_delta_U'][i]['dtid'] if selected['full_delta_U'][i] else 0)
                for i, row in policy.items())
        with (folder/(label+'_oracle_choices.csv')).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(choice_records[0])); writer.writeheader(); writer.writerows(choice_records)
        assert result['policies']['full_delta_U']['harmful_proxy_images'] == 0
        for p in paths: assert sha(ROOT/p) == INPUT_HASHES[p], 'Input changed during probe'
        report['pairs'][label] = result; save(report, folder/'report.json')
        print('PAIR_COMPLETE '+label+' '+json.dumps(result), flush=True)
    for path, expected in INPUT_HASHES.items():
        assert sha(ROOT/path) == expected, 'Frozen input changed by completion: '+path
    report['status'] = 'complete'; save(report, folder/'report.json')


def fingerprint():
    paths = [ENV/'bin/python']+[ENV/('lib/python3.7/site-packages/pycocotools/'+name)
                              for name in ('coco.py', 'cocoeval.py', 'mask.py')]
    for pattern in ('lib/python3.7/site-packages/numpy/core/_multiarray_umath*.so',
                    'lib/python3.7/site-packages/pycocotools/_mask*.so'):
        matches = list(ENV.glob(pattern)); assert len(matches) == 1, 'Ambiguous native binary: '+pattern
        paths.extend(matches)
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true'); p.add_argument('--launch', action='store_true')
    p.add_argument('--worker', action='store_true'); p.add_argument('--output-dir', type=Path)
    a = p.parse_args(); os.chdir(str(ROOT))
    if a.self_test: self_test(); return
    folder = a.output_dir.resolve(); folder.relative_to(ROOT/'output_offline')
    if a.launch:
        if folder.exists(): raise RuntimeError('New output required')
        source_hashes = {f: sha(ROOT/f) for f in SOURCES}; runtime = fingerprint()
        folder.mkdir(parents=True); cpus = sorted(os.sched_getaffinity(0))[:2]; env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            LD_LIBRARY_PATH=str(ENV/'lib')+':'+str(ENV/'lib/python3.7/site-packages/torch/lib'))
        manifest = dict(source_sha256=source_hashes, runtime_sha256=runtime, input_sha256=INPUT_HASHES,
                        cpu_affinity=cpus, gpu_disabled=True, test_split_access=False)
        save(manifest, folder/'deployment.json')
        cmd = ['ionice', '-c', '2', '-n', '7', 'nice', '-n', '10', 'taskset', '-c', ','.join(map(str, cpus)),
               str(ENV/'bin/python'), '-u', __file__, '--worker', '--output-dir', str(folder)]
        with (folder/'console.log').open('x') as log:
            child = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        manifest['pid'] = child.pid; save(manifest, folder/'deployment.json'); print(json.dumps(manifest)); return
    if not a.worker: raise RuntimeError('Use --self-test or --launch')
    code, error = 0, None
    try:
        manifest = json.loads((folder/'deployment.json').read_text())
        assert Path(sys.prefix).resolve() == ENV.resolve()
        assert all(sha(ROOT/f) == h for f, h in manifest['source_sha256'].items())
        assert fingerprint() == manifest['runtime_sha256']
        probe(folder)
    except BaseException as exc:
        code, error = 1, repr(exc); raise
    finally:
        save(dict(exit_code=code, error=error, finished_unix=time.time()), folder/'completion.json')


if __name__ == '__main__': main()
