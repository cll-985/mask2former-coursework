"""Frozen binary-only image-block crossfit diagnostic; exploratory, NOT confirmation."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.mva_geometry_probe import ENV, GT, BASES, RCNN, sha, save

SPEC = '.aris/compute/mva-region-binary-crossfit-20261103.md'
PLAN = 'refine-logs/MVA_REGION_NEXT_DECISION_20261103.md'
SOURCES = ['offline_tools/mva_region_binary_crossfit.py', 'offline_tools/mva_region_binary_features.py',
    'offline_tools/mva_region_binary_fit.py', 'offline_tools/mva_region_ap_probe.py',
    'offline_tools/mva_region_capacity_probe.py', 'offline_tools/mva_geometry_probe.py',
    'offline_tools/mva_fusion_controls.py', 'offline_tools/compare_fusion_rules.py',
    'offline_tools/fuse_coco_instances.py', 'offline_tools/mva_full_coco_bootstrap.py', SPEC, PLAN,
    '.aris/compute/mva-region-ap-probe-20261103.md', '.aris/compute/mva-region-capacity-probe-20261102.md',
    '.aris/compute/mva-geometry-probe-20261101.md', '.aris/compute/mva-arch-analysis-20261030.md']
MATCH_COUNTS = {'mask2former_42_3407': 6256, 'maskrcnn_42_3407': 5413}
ANCHOR_COUNTS = {'mask2former_42_3407': 3115, 'maskrcnn_42_3407': 2808}
STARTED = None


def guard():
    if STARTED is not None and time.monotonic()-STARTED > 7200: raise TimeoutError('Two-hour wall cap')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 12*1024**2:
        raise MemoryError('Own high-water RSS above12GiB')


def anchor(coco, annotations, a, b):
    import numpy as np
    from pycocotools import mask as mu
    from offline_tools.fuse_coco_instances import rle_for_compute
    if not annotations: return None, None
    endpoint = a if float(a['score']) >= float(b['score']) else b
    targets = [coco.annToRLE(x) for x in annotations]
    overlaps = mu.iou([rle_for_compute(endpoint['segmentation'])], targets, [0]*len(targets))[0]
    index = int(np.argmax(overlaps))
    if float(overlaps[index]) <= 0: return None, None
    return targets[index], int(annotations[index]['id'])


def decoded_features(a, b, category):
    from pycocotools import mask as mu
    from offline_tools.fuse_coco_instances import rle_for_compute
    from offline_tools.mva_region_binary_features import extract
    masks = mu.decode([rle_for_compute(x['segmentation']) for x in (a, b)]).astype(bool)
    return extract(masks[:, :, 0], masks[:, :, 1], float(a['score']), float(b['score']), int(category))


def collect(dataset, outputs, provenance, folds):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    from offline_tools.mva_region_binary_features import target, reconstruct, whole_reconstruct
    coco = COCO(); coco.dataset = copy.deepcopy(dataset); coco.createIndex()
    targets = defaultdict(list)
    for ann in dataset['annotations']:
        if not ann.get('iscrowd', 0): targets[(ann['image_id'], ann['category_id'])].append(ann)
    rx, ry, ra, ri, rc, wx, wy, wi, wc, candidates = [], [], [], [], [], [], [], [], [], []
    offset = 0
    for dtid, (item, sources) in enumerate(zip(outputs, provenance), 1):
        if sources is None: continue
        a, b = sources
        data = decoded_features(a, b, item['category_id'])
        k = len(data['areas']); image = int(item['image_id'])
        # Extract before consulting GT; all matched candidates remain represented.
        frozen_r, frozen_w = data['region_features'].copy(), data['whole_features'].copy()
        truth_rle, gt_id = anchor(coco, targets[(image, int(item['category_id']))], a, b)
        truth = mu.decode(truth_rle).astype(bool) if truth_rle is not None else None
        bits, menu_index = target(data, truth)
        assert np.array_equal(frozen_r, data['region_features']) and np.array_equal(frozen_w, data['whole_features'])
        row = dict(dtid=dtid, image_id=image, fold=folds[image], category_id=int(item['category_id']),
            component_count=k, start=offset, stop=offset+k, anchor_gt_id=gt_id, truth_rle=truth_rle,
            target_whole=int(menu_index))
        if truth is not None:
            def iou(m): return int((m & truth).sum())/int((m | truth).sum())
            row['target_region_iou'] = iou(reconstruct(data, bits))
            row['target_whole_iou'] = iou(whole_reconstruct(data, menu_index))
        candidates.append(row)
        if k:
            rx.append(data['region_features']); ry.append(np.asarray(bits, dtype=int)); ra.append(data['areas'])
            ri.extend([image]*k); rc.extend([dtid]*k); offset += k
            wx.append(data['whole_features']); wy.append(int(menu_index)); wi.append(image); wc.append(dtid)
        if len(candidates) % 250 == 0:
            guard(); print('FEATURE_PROGRESS candidates={} components={}'.format(len(candidates), offset), flush=True)
    assert offset > 0 and len(wx) > 0
    return dict(rx=np.concatenate(rx).astype(np.float64), ry=np.concatenate(ry), ra=np.concatenate(ra),
        ri=np.asarray(ri, dtype=int), rc=np.asarray(rc, dtype=int), wx=np.asarray(wx, dtype=np.float64),
        wy=np.asarray(wy, dtype=int), wi=np.asarray(wi, dtype=int), wc=np.asarray(wc, dtype=int), candidates=candidates)


def crossfit(data, folds, folder, label):
    import numpy as np
    from offline_tools.mva_region_binary_fit import fit, predict, weights
    probabilities = np.full(len(data['ry']), np.nan); menus = np.full((len(data['wy']), 4), np.nan)
    logs = []
    for fold in range(5):
        for name in ('region', 'whole'):
            region = name == 'region'; prefix = 'r' if region else 'w'
            x, y, images, candidates = [data[prefix+s] for s in ('x', 'y', 'i', 'c')]
            test = np.asarray([folds[int(i)] == fold for i in images]); train = ~test
            assert test.any() and train.any()
            train_ids, test_ids = set(map(int, images[train])), set(map(int, images[test]))
            assert not (train_ids & test_ids)
            w = weights(images[train], candidates[train], data['ra'][train] if region else None)
            model = fit(x[train], y[train], w, 18 if region else 22, 2 if region else 4)
            proba = predict(model, x[test])
            if region: probabilities[test] = proba
            else: menus[test] = proba
            record = dict(fold=fold, model=name, train_images=sorted(train_ids), test_images=sorted(test_ids),
                          train_rows=int(train.sum()), test_rows=int(test.sum()), parameters=model)
            save(record, folder/('{}_{}_fold{}.json'.format(label, name, fold)))
            logs.append({k: v for k, v in record.items() if k != 'parameters'})
            print('FIT_COMPLETE {} {} fold={} train={} test={}'.format(label, name, fold, int(train.sum()), int(test.sum())), flush=True)
            guard()
    assert np.isfinite(probabilities).all() and np.isfinite(menus).all()
    assert np.allclose(menus.sum(axis=1), 1)
    np.savez_compressed(str(folder/(label+'_oof.npz')), region_probability=probabilities, whole_probability=menus)
    return probabilities, menus, logs


def assemble(outputs, provenance, data, probabilities, menus, folds, image_ids):
    import numpy as np
    from pycocotools import mask as mu
    from offline_tools.fuse_coco_instances import rle_for_compute
    from offline_tools.mva_region_binary_features import reconstruct, whole_reconstruct
    from offline_tools.mva_region_ap_probe import encode
    systems = {name: copy.deepcopy(outputs) for name in ('region', 'whole')}
    wm = {int(d): i for i, d in enumerate(data['wc'])}; rows = []
    for meta in data['candidates']:
        dtid = meta['dtid']; item = outputs[dtid-1]; a, b = provenance[dtid-1]
        d = decoded_features(a, b, item['category_id']); k = len(d['areas'])
        assert k == meta['component_count']
        original = mu.decode(rle_for_compute(item['segmentation'])).astype(bool)
        assert np.array_equal(reconstruct(d, d['base_bits']), original), 'Default mask reconstruction differs from legacy endpoint'
        row = {key: value for key, value in meta.items() if key != 'truth_rle'}
        if k:
            sl = slice(meta['start'], meta['stop'])
            assert np.array_equal(d['region_features'], data['rx'][sl])
            assert np.array_equal(d['whole_features'], data['wx'][wm[dtid]])
            p = probabilities[sl]; bits = np.where(p == .5, d['base_bits'], p > .5)
            index = int(np.argmax(menus[wm[dtid]]))
            masks = dict(region=reconstruct(d, bits), whole=whole_reconstruct(d, index))
            row['region_kept_components'] = int(bits.sum()); row['whole_choice'] = index
        else:
            masks = dict(region=original, whole=original); row['region_kept_components'] = 0; row['whole_choice'] = -1
        truth = mu.decode(meta['truth_rle']).astype(bool) if meta['truth_rle'] is not None else None
        for name, mask in masks.items():
            row[name+'_changed_mask'] = int(not np.array_equal(mask, original))
            rle = encode(mask); systems[name][dtid-1]['segmentation'] = rle
            systems[name][dtid-1]['bbox'] = mu.toBbox(rle_for_compute(rle)).tolist()
            if truth is not None:
                value = int((mask & truth).sum())/int((mask | truth).sum())
                row[name+'_iou'] = value; row[name+'_regret'] = meta['target_'+name+'_iou']-value
                assert row[name+'_regret'] >= -1e-12, 'Teacher target not optimal in its local action menu'
        rows.append(row)
        if len(rows) % 250 == 0: guard()
    per_image = defaultdict(list)
    for row in rows:
        if row['anchor_gt_id'] is not None: per_image[row['image_id']].append(row['region_iou']-row['whole_iou'])
    local = []
    for fold in range(5):
        ids = [i for i in image_ids if folds[i] == fold]
        values = [float(np.mean(per_image[i])) if per_image[i] else 0.0 for i in ids]
        local.append(dict(fold=fold, images=len(ids), region_minus_whole_image_equal_iou=float(np.mean(values)) if values else 0.0))
    for values in systems.values():
        assert len(values) == len(outputs)
        assert all(all(a[k] == b[k] for k in ('image_id', 'category_id', 'score')) for a, b in zip(outputs, values))
    return systems, rows, local


def probe(folder):
    import numpy as np
    from offline_tools.mva_region_capacity_probe import INPUT_HASHES
    from offline_tools.mva_region_ap_probe import templates, full_metrics, EXPECTED
    from offline_tools.mva_fusion_controls import binary_geometry_control
    from offline_tools.mva_region_binary_fit import make_folds
    from offline_tools.mva_region_binary_features import FEATURE_SCHEMA
    for path, h in INPUT_HASHES.items(): assert sha(ROOT/path) == h
    dataset = json.loads((ROOT/GT).read_text()); image_ids = sorted(int(x['id']) for x in dataset['images'])
    group_fields = [k for k in ('vehicle_id', 'sequence_id', 'video_id', 'group_id') if any(k in x for x in dataset['images'])]
    assert not group_fields, 'Known grouping requires grouped protocol: '+str(group_fields)
    assert len(image_ids) == 810 and len(set(image_ids)) == 810
    folds = make_folds(image_ids); save({str(k): v for k, v in folds.items()}, folder/'fold_manifest.json')
    save(FEATURE_SCHEMA, folder/'feature_schema.json')
    report = dict(status='running', exploratory=True, gpu_disabled=True, test_split_access=False,
        scope='image-block OOF CPU supervised diagnostic on already studied val; not independent confirmation or novel method evidence',
        folds=5, missing_vehicle_group_metadata=True, input_sha256=INPUT_HASHES,
        geometry_only=True, frozen_plan_sha256=sha(ROOT/PLAN), pairs={})
    for label, paths in [('mask2former_42_3407', [BASES['42'], BASES['3407']]), ('maskrcnn_42_3407', RCNN)]:
        raw = [json.loads((ROOT/p).read_text()) for p in paths]
        outputs, provenance = templates(raw[0], raw[1], 'endpoint')
        reference = [x for x in binary_geometry_control(raw[0], raw[1], 'endpoint')[0] if float(x['score']) >= .001]
        assert outputs == reference
        endpoint = full_metrics(dataset, outputs)
        assert abs(endpoint['AP']-EXPECTED[label][0]) < 1e-6
        data = collect(dataset, outputs, provenance, folds)
        assert len(data['candidates']) == MATCH_COUNTS[label]
        assert sum(x['anchor_gt_id'] is not None for x in data['candidates']) == ANCHOR_COUNTS[label]
        np.savez_compressed(str(folder/(label+'_features.npz')), **{k: v for k, v in data.items() if k != 'candidates'})
        save([{k: v for k, v in row.items() if k != 'truth_rle'} for row in data['candidates']], folder/(label+'_candidate_manifest.json'))
        p, m, fits = crossfit(data, folds, folder, label)
        systems, rows, local = assemble(outputs, provenance, data, p, m, folds, image_ids)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (folder/(label+'_decisions.csv')).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        result = dict(endpoint=endpoint, matched=len(rows), component_rows=len(data['ry']),
            no_anchor=sum(x['anchor_gt_id'] is None for x in rows), zero_xor=sum(x['component_count'] == 0 for x in rows),
            fits=fits, local_by_fold=local, models={})
        for name, values in systems.items():
            path = folder/(label+'_'+name+'_oof.json'); save(values, path)
            metrics = full_metrics(dataset, values)
            edits = sum(row[name+'_changed_mask'] for row in rows)
            negatives = [row for row in rows if row['anchor_gt_id'] is None]
            metrics.update(output_sha256=sha(path), changed_masks=edits,
                no_anchor_changed_masks=sum(row[name+'_changed_mask'] for row in negatives),
                no_anchor_changed_fraction=sum(row[name+'_changed_mask'] for row in negatives)/len(negatives) if negatives else None,
                delta_endpoint_AP=metrics['AP']-endpoint['AP'])
            result['models'][name] = metrics
            print('OOF_COCO_COMPLETE {} {} AP={:.8f}'.format(label, name, metrics['AP']), flush=True); guard()
        result['region_minus_whole_AP'] = result['models']['region']['AP']-result['models']['whole']['AP']
        result['positive_local_folds'] = sum(x['region_minus_whole_image_equal_iou'] > 0 for x in local)
        result['pair_gate'] = (result['region_minus_whole_AP'] >= .20 and result['models']['region']['AP'] > endpoint['AP']
                               and result['positive_local_folds'] >= 4)
        report['pairs'][label] = result; save(report, folder/'report.json')
        for path, h in INPUT_HASHES.items(): assert sha(ROOT/path) == h
    report['m2f_strong_baseline_AP'] = 56.079893043726535
    report['exploratory_gate_passed'] = (all(v['pair_gate'] for v in report['pairs'].values()) and
        report['pairs']['mask2former_42_3407']['models']['region']['AP'] > report['m2f_strong_baseline_AP'])
    report['status'] = 'complete'; save(report, folder/'report.json')


def self_test():
    import numpy as np
    from offline_tools import mva_region_binary_features as f, mva_region_binary_fit as m
    from offline_tools.mva_region_ap_probe import templates, encode
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path(sys.prefix).resolve() == ENV.resolve()
    print(json.dumps(dict(feature_witness=f.self_test(), fit_witness=m.self_test())), flush=True)
    a = np.zeros((9, 11), dtype=bool); a[2:5, 2:5] = True; b = a.copy(); b[5, 3] = True
    def item(image, category, score, mask): return dict(image_id=image, category_id=category, score=score, segmentation=encode(mask))
    dataset = dict(images=[dict(id=i, height=9, width=11) for i in range(1, 4)],
        categories=[dict(id=1, name='x')], annotations=[dict(id=1, image_id=1, category_id=1,
        segmentation=encode(a), area=9, bbox=[2, 2, 3, 3], iscrowd=0)])
    left = [item(1, 1, .8, a), item(2, 1, .8, a), item(3, 1, .8, a)]
    right = [item(1, 1, .7, b), item(2, 1, .8, b), item(3, 1, .7, a)]
    baseline, provenance = templates(left, right, 'endpoint'); folds = m.make_folds([1, 2, 3])
    data = collect(dataset, baseline, provenance, folds)
    assert len(data['candidates']) == 3 and sum(x['anchor_gt_id'] is not None for x in data['candidates']) == 1
    assert sum(x['component_count'] == 0 for x in data['candidates']) == 1
    assert data['rx'].shape[1] == 24 and data['wx'].shape[1] == 28
    # p=.5 must reconstruct baseline even on no-anchor candidates; whole targets reproduce teacher.
    menu = np.eye(4)[data['wy']]
    systems, rows, local = assemble(baseline, provenance, data, np.full(len(data['ry']), .5), menu, folds, [1, 2, 3])
    assert all(row['region_changed_mask'] == 0 for row in rows)
    assert sum(x['component_count'] == 0 for x in rows) == 1
    assert all(x['whole_changed_mask'] == 0 for x in rows if x['anchor_gt_id'] is None)
    assert len(local) == 5  # Empty folds only in this tiny synthetic fixture; handled explicitly below.
    print(json.dumps(dict(status='passed', all_matched_fixture=3, no_anchor_fixture=2, exact_tie=True,
        empty_xor=True, default_reconstruction=True, no_gt_feature_gate=True)), flush=True)


def fingerprint():
    from offline_tools.mva_region_ap_probe import fingerprint as previous
    result = previous(); site = ENV/'lib/python3.7/site-packages'
    for package in ('scipy/optimize', 'scipy/special'):
        for path in sorted((site/package).glob('*')):
            if path.is_file() and (path.suffix == '.py' or path.suffix == '.so'):
                result[str(path.relative_to(ROOT))] = sha(path)
    return result


def main():
    global STARTED
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true'); p.add_argument('--launch', action='store_true')
    p.add_argument('--worker', action='store_true'); p.add_argument('--output-dir', type=Path)
    args = p.parse_args(); os.chdir(str(ROOT))
    if args.self_test: self_test(); return
    folder = args.output_dir.resolve(); folder.relative_to(ROOT/'output_offline')
    if args.launch:
        if folder.exists(): raise RuntimeError('Exclusive new output required')
        from offline_tools.mva_region_capacity_probe import INPUT_HASHES
        manifest = dict(source_sha256={f: sha(ROOT/f) for f in SOURCES}, runtime_sha256=fingerprint(),
            input_sha256=INPUT_HASHES, gpu_disabled=True, test_split_access=False, wall_cap_seconds=7200,
            started_unix=time.time(), cpu_affinity=sorted(os.sched_getaffinity(0))[:2])
        env = dict(os.environ); env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            LD_LIBRARY_PATH=str(ENV/'lib')+':'+str(ENV/'lib/python3.7/site-packages/torch/lib'))
        folder.mkdir(parents=True); save(manifest, folder/'deployment.json')
        cmd = ['ionice', '-c', '2', '-n', '7', 'nice', '-n', '10', 'taskset', '-c',
            ','.join(map(str, manifest['cpu_affinity'])), str(ENV/'bin/python'), '-u', __file__, '--worker', '--output-dir', str(folder)]
        with (folder/'console.log').open('x') as log:
            child = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        manifest['pid'] = child.pid; save(manifest, folder/'deployment.json'); print(json.dumps(manifest)); return
    if not args.worker: raise RuntimeError('Use --self-test or --launch')
    code, error = 0, None; STARTED = time.monotonic()
    def timed_out(signum, frame): raise TimeoutError('Two-hour wall cap')
    signal.signal(signal.SIGALRM, timed_out); signal.alarm(7200)
    try:
        manifest = json.loads((folder/'deployment.json').read_text())
        assert Path(sys.prefix).resolve() == ENV.resolve() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        assert all(sha(ROOT/f) == h for f, h in manifest['source_sha256'].items())
        assert fingerprint() == manifest['runtime_sha256']; probe(folder)
        assert all(sha(ROOT/f) == h for f, h in manifest['source_sha256'].items())
        assert fingerprint() == manifest['runtime_sha256']
    except BaseException as exc:
        code, error = 1, repr(exc); raise
    finally:
        signal.alarm(0)
        save(dict(exit_code=code, error=error, finished_unix=time.time(), elapsed_seconds=time.monotonic()-STARTED), folder/'completion.json')
        hashes = {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()
                  and p.name not in ('output_hashes.json', 'console.log') and not p.name.endswith('.tmp')}
        save(dict(sha256=hashes, exclusions=['self-referential output_hashes.json', 'live console.log; hash after process exits']),
             folder/'output_hashes.json')


if __name__ == '__main__': main()
