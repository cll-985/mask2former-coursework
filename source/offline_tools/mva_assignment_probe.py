"""Exploratory fixed-order COCO matching replay; no training or AP guarantee."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.mva_geometry_probe import ROOT, ENV, GT, EXPECTED_GT, BASES, RCNN, sha, save

SPEC = '.aris/compute/mva-assignment-probe-20261102.md'
SOURCES = ['offline_tools/mva_assignment_probe.py', 'offline_tools/mva_geometry_probe.py',
           'offline_tools/mva_fusion_controls.py', 'offline_tools/compare_fusion_rules.py',
           'offline_tools/fuse_coco_instances.py', 'offline_tools/mva_full_coco_bootstrap.py', SPEC]


def transitions(left, right, changed):
    import numpy as np
    assert left.params.iouThrs.tolist() == right.params.iouThrs.tolist()
    records = []
    totals = [dict(threshold=float(t), tp_before=0, tp_after=0, gained_tp=0,
                   lost_tp=0, reassigned_tp=0, lost_with_nondecreasing_max_iou=0,
                   tp_changed_on_unedited_detection=0) for t in left.params.iouThrs]
    for a, b in zip(left.evalImgs, right.evalImgs):
        assert (a is None) == (b is None)
        if a is None or a['aRng'] != left.params.areaRng[0]:
            continue
        assert a['image_id'] == b['image_id'] and a['category_id'] == b['category_id']
        assert a['dtIds'] == b['dtIds'] and a['dtScores'] == b['dtScores']
        assert np.array_equal(a['gtIgnore'], b['gtIgnore']) and a['gtIds'] == b['gtIds']
        assert np.array_equal(a['dtIgnore'], b['dtIgnore']), 'All-area ignore changed'
        good = ~a['dtIgnore'].astype(bool)
        old = (a['dtMatches'] > 0) & good
        new = (b['dtMatches'] > 0) & good
        key = (a['image_id'], a['category_id'])
        ia, ib = np.asarray(left.ious[key]), np.asarray(right.ious[key])
        # Only descriptive maximum-overlap comparison; not GT identity tracking.
        gts = left._gts[key]
        eligible = [i for i, g in enumerate(gts) if not g.get('ignore', 0)]
        nondecreasing = np.zeros(len(a['dtIds']), dtype=bool)
        if ia.size and ib.size and eligible:
            nondecreasing = ib[:, eligible].max(axis=1) >= ia[:, eligible].max(axis=1)-1e-12
        edited = np.asarray([changed[x] for x in a['dtIds']], dtype=bool)
        for i, total in enumerate(totals):
            gain, loss = ~old[i] & new[i], old[i] & ~new[i]
            row = dict(image_id=a['image_id'], category_id=a['category_id'],
                       threshold=total['threshold'], tp_before=int(old[i].sum()),
                       tp_after=int(new[i].sum()), gained_tp=int(gain.sum()), lost_tp=int(loss.sum()),
                       reassigned_tp=int((old[i] & new[i] & (a['dtMatches'][i] != b['dtMatches'][i])).sum()),
                       lost_with_nondecreasing_max_iou=int((loss & nondecreasing).sum()),
                       tp_changed_on_unedited_detection=int(((gain | loss) & ~edited).sum()))
            records.append(row)
            for name in total:
                if name != 'threshold':
                    total[name] += row[name]
    for total in totals:
        assert total['tp_after']-total['tp_before'] == total['gained_tp']-total['lost_tp']
    return totals, records


def self_test():
    import numpy as np
    from pycocotools import mask as mu
    from offline_tools.mva_full_coco_bootstrap import evaluate
    mask = np.zeros((8, 8), dtype=np.uint8); mask[:2, :4] = 1
    expanded = mask.copy(); expanded[2:4, :4] = 1
    def rle(m):
        r = mu.encode(np.asfortranarray(m)); r['counts'] = r['counts'].decode('ascii'); return r
    dataset = dict(images=[dict(id=1, height=8, width=8)], categories=[dict(id=1, name='x')],
                   annotations=[dict(id=1, image_id=1, category_id=1, segmentation=rle(mask),
                                     area=8, bbox=[0, 0, 4, 2], iscrowd=0)])
    pred = [dict(image_id=1, category_id=1, segmentation=rle(mask), score=.9)]
    new = [dict(image_id=1, category_id=1, segmentation=rle(expanded), score=.9)]
    a, b = evaluate(dataset, pred), evaluate(dataset, new)
    zero, _ = transitions(a, a, {1: False})
    assert all(x['gained_tp'] == x['lost_tp'] == 0 for x in zero)
    changes, _ = transitions(a, b, {1: True})
    assert changes[0]['tp_before'] == changes[0]['tp_after'] == 1
    assert all(x['lost_tp'] == 1 and x['tp_after'] == 0 for x in changes[1:])
    duplicate = pred + [dict(pred[0], score=.8)]
    c = evaluate(dataset, duplicate)
    dup, _ = transitions(c, c, {1: False, 2: False})
    assert all(x['tp_before'] == 1 for x in dup)
    empty = evaluate(dataset, [])
    z, _ = transitions(empty, empty, {})
    assert all(x['tp_before'] == x['tp_after'] == 0 for x in z)
    print(json.dumps(dict(status='passed', native_coco_fixtures=4, thresholds_per_fixture=10)), flush=True)


def probe(folder):
    import numpy as np
    from offline_tools.mva_fusion_controls import binary_geometry_control
    from offline_tools.mva_full_coco_bootstrap import evaluate
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or sha(ROOT/GT) != EXPECTED_GT:
        raise RuntimeError('Runtime/data contract mismatch')
    dataset = json.loads((ROOT/GT).read_text())
    pairs = [('mask2former_'+a+'_'+b, [BASES[a], BASES[b]])
             for a, b in [('42', '3407'), ('42', '2026'), ('3407', '2026')]]
    pairs.append(('maskrcnn_42_3407', RCNN))
    report = dict(status='running', exploratory=True, gpu_disabled=True, test_split_access=False,
                  scope='simultaneous all-union vs endpoint intervention; NOT single-edit marginal utility or new confirmatory AP',
                  ranking='same matches/scores/stable order/floor .001/top100; endpoint uses union for exact score ties per frozen implementation',
                  gt_sha256=EXPECTED_GT, pairs={})
    for label, paths in pairs:
        hashes = {p: sha(ROOT/p) for p in paths}
        raw = [json.loads((ROOT/p).read_text()) for p in paths]
        outputs = [binary_geometry_control(raw[0], raw[1], kind)[0] for kind in ('endpoint', 'union')]
        outputs = [[v for v in values if float(v['score']) >= .001] for values in outputs]
        assert len(outputs[0]) == len(outputs[1])
        changed = {}
        for index, (a, b) in enumerate(zip(*outputs), 1):
            assert all(a[k] == b[k] for k in ('image_id', 'category_id', 'score'))
            changed[index] = a['segmentation'] != b['segmentation']
        ev = [evaluate(dataset, x) for x in outputs]
        aps = []
        for e in ev:
            p = e.eval['precision'][:, :, :, 0, -1]; aps.append(float(p[p > -1].mean()*100))
        totals, rows = transitions(ev[0], ev[1], changed)
        with (folder/(label+'.csv')).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        result = dict(source_sha256=hashes, candidate_count=len(outputs[0]), changed_geometry=sum(changed.values()),
                      endpoint_AP=aps[0], union_AP=aps[1], delta_AP=aps[1]-aps[0], thresholds=totals)
        assert all(sha(ROOT/p) == h for p, h in hashes.items()), 'Input changed'
        report['pairs'][label] = result; save(report, folder/'report.json')
        print('PAIR_COMPLETE '+label+' '+json.dumps(result), flush=True)
    report['status'] = 'complete'; save(report, folder/'report.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true'); p.add_argument('--launch', action='store_true')
    p.add_argument('--worker', action='store_true'); p.add_argument('--output-dir', type=Path)
    a = p.parse_args(); os.chdir(str(ROOT))
    if a.self_test:
        self_test(); return
    folder = a.output_dir.resolve(); folder.relative_to(ROOT/'output_offline')
    if a.launch:
        if folder.exists(): raise RuntimeError('New output required')
        folder.mkdir(parents=True); cpus = sorted(os.sched_getaffinity(0))[:2]; env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   LD_LIBRARY_PATH=str(ENV/'lib')+':'+str(ENV/'lib/python3.7/site-packages/torch/lib'))
        manifest = dict(source_sha256={f: sha(ROOT/f) for f in SOURCES}, cpu_affinity=cpus,
                        gpu_disabled=True, test_split_access=False)
        save(manifest, folder/'deployment.json')
        cmd = ['ionice', '-c', '2', '-n', '7', 'nice', '-n', '10', 'taskset', '-c',
               ','.join(map(str, cpus)), str(ENV/'bin/python'), '-u', __file__, '--worker', '--output-dir', str(folder)]
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
        probe(folder)
    except BaseException as exc:
        code, error = 1, repr(exc); raise
    finally:
        save(dict(exit_code=code, error=error, finished_unix=time.time()), folder/'completion.json')


if __name__ == '__main__': main()
