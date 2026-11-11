"""Exploratory GT-assisted geometry policies evaluated by FULL COCO; no AP bound."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.mva_geometry_probe import ENV, GT, BASES, RCNN, sha, save
from offline_tools.mva_region_capacity_probe import INPUT_HASHES, best_component_prefix, membership_components

SPEC = '.aris/compute/mva-region-ap-probe-20261103.md'
SOURCES = ['offline_tools/mva_region_ap_probe.py', 'offline_tools/mva_region_capacity_probe.py',
    'offline_tools/mva_geometry_probe.py', 'offline_tools/mva_fusion_controls.py',
    'offline_tools/compare_fusion_rules.py', 'offline_tools/fuse_coco_instances.py',
    'offline_tools/mva_full_coco_bootstrap.py', SPEC,
    '.aris/compute/mva-region-capacity-probe-20261102.md',
    '.aris/compute/mva-geometry-probe-20261101.md', '.aris/compute/mva-arch-analysis-20261030.md']
POLICIES = ('endpoint', 'union', 'whole', 'component', 'pixel')
EXPECTED = {'mask2former_42_3407': (55.3292597933276, 54.8284670411996),
            'maskrcnn_42_3407': (52.574541199563285, 53.28841564746991)}


def encode(mask):
    import numpy as np
    from pycocotools import mask as mu
    result = mu.encode(np.asfortranarray(mask, dtype=np.uint8))
    result['counts'] = result['counts'].decode('ascii')
    return result


def assembled_masks(a, b, truth):
    """Return local-IoU-max masks to ONE chosen GT; not COCO-AP optimal masks."""
    import numpy as np
    a, b, truth = [np.asarray(x, dtype=bool) for x in (a, b, truth)]
    assert a.shape == b.shape == truth.shape and a.ndim == 2
    g = int(truth.sum()); assert g > 0
    full_union = a | b
    if not full_union.any():
        return {k: np.zeros(a.shape, dtype=bool) for k in ('whole', 'component', 'pixel')}
    yy, xx = np.nonzero(full_union)
    box = np.s_[int(yy.min()):int(yy.max())+1, int(xx.min()):int(xx.max())+1]
    aa, bb, gg = [x[box] for x in (a, b, truth)]
    core, union = aa & bb, aa | bb
    choices = (aa, bb, union, core)
    ratios = [(int((m & gg).sum()), g+int((m & ~gg).sum())) for m in choices]
    best = 0
    for i in range(1, 4):
        if ratios[i][0]*ratios[best][1] > ratios[best][0]*ratios[i][1]: best = i
    stats, labels = membership_components(aa, bb, gg)
    cn, cd, selected = best_component_prefix(int((core & gg).sum()), g+int((core & ~gg).sum()), stats)
    component = core.copy(); offset = 0
    for lab in labels:
        count = int(lab.max())
        local_ids = [i-offset+1 for i in selected if offset <= i < offset+count]
        component |= np.isin(lab, local_ids)
        offset += count
    pixel = core | ((aa ^ bb) & gg)
    n, d = int((component & gg).sum()), g+int((component & ~gg).sum())
    assert n*cd == cn*d, 'Selected labels disagree with exact component optimum'
    wn, wd = ratios[best]; pn, pd = int((pixel & gg).sum()), g+int((pixel & ~gg).sum())
    assert wn*cd <= cn*wd and cn*pd <= pn*cd
    result = {}
    for name, crop in zip(('whole', 'component', 'pixel'), (choices[best], component, pixel)):
        value = np.zeros(a.shape, dtype=bool); value[box] = crop
        assert np.all((a & b) <= value) and np.all(value <= full_union)
        result[name] = value
    return result


def templates(base, other, kind):
    """Exact frozen control plus source indices; geometry is edited AFTER top100."""
    from pycocotools import mask as mu
    from offline_tools.compare_fusion_rules import group_results, prune_topk
    from offline_tools.fuse_coco_instances import greedy_matches, rle_for_compute
    groups = [group_results(values) for values in (base, other)]
    candidates, sources = [], {}
    for key in sorted(set(groups[0]) | set(groups[1])):
        left, right = [sorted(group[key], key=lambda x: -float(x['score'])) for group in groups]
        used = [set(), set()]
        for i, j, _ in greedy_matches(left, right, .5):
            a, b = left[i], right[j]; used[0].add(i); used[1].add(j)
            item = copy.deepcopy(a if float(a['score']) >= float(b['score']) else b)
            if kind == 'union' or float(a['score']) == float(b['score']):
                r = mu.merge([rle_for_compute(x['segmentation']) for x in (a, b)])
                item['bbox'] = mu.toBbox(r).tolist(); r['counts'] = r['counts'].decode('ascii')
                item['segmentation'] = r
            item['score'] = (float(a['score'])+float(b['score']))/2
            item['_source_key'] = (key[0], key[1], i, j)
            sources[item['_source_key']] = (a, b)
            candidates.append(item)
        for values, indices in zip((left, right), used):
            for i, source in enumerate(values):
                if i not in indices:
                    item = copy.deepcopy(source); item['score'] = float(source['score'])/2
                    candidates.append(item)
    kept = [x for x in prune_topk(candidates, 100) if float(x['score']) >= .001]
    provenance, clean = [], []
    for item in kept:
        key = item.pop('_source_key', None)
        provenance.append(sources.get(key)); clean.append(item)
    return clean, provenance


def full_metrics(dataset, predictions):
    import numpy as np
    from offline_tools.mva_full_coco_bootstrap import evaluate
    # COCO mutates annotations; every evaluation gets separate deep copies.
    ev = evaluate(copy.deepcopy(dataset), copy.deepcopy(predictions))
    def mean(array):
        valid = array[array > -1]
        return float(valid.mean()*100) if valid.size else None
    p, r = ev.eval['precision'], ev.eval['recall']
    result = dict(AP=mean(p[:, :, :, 0, -1]), AP50=mean(p[0:1, :, :, 0, -1]),
        AP75=mean(p[5:6, :, :, 0, -1]), APs=mean(p[:, :, :, 1, -1]),
        APm=mean(p[:, :, :, 2, -1]), APl=mean(p[:, :, :, 3, -1]), AR100=mean(r[:, :, 0, -1]))
    names = {int(x['id']): x['name'] for x in dataset['categories']}
    result['per_category_AP'] = {str(k): dict(name=names[k], AP=mean(p[:, :, i, 0, -1]))
                                 for i, k in enumerate(ev.params.catIds)}
    result['tp_by_threshold'] = [0]*len(ev.params.iouThrs)
    for record in ev.evalImgs:
        if record is not None and record['aRng'] == ev.params.areaRng[0]:
            tp = (record['dtMatches'] > 0) & ~record['dtIgnore'].astype(bool)
            result['tp_by_threshold'] = [x+int(y) for x, y in zip(result['tp_by_threshold'], tp.sum(axis=1))]
    assert np.allclose(ev.params.iouThrs, np.linspace(.5, .95, 10))
    return result


def self_test():
    import numpy as np
    from pycocotools import mask as mu
    from offline_tools.mva_region_capacity_probe import capacity
    from offline_tools.mva_fusion_controls import binary_geometry_control
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path(sys.prefix).resolve() == ENV.resolve()
    rng = np.random.RandomState(20261103)
    for _ in range(64):
        a, b, g = [rng.rand(9, 11) < .35 for i in range(3)]; g[8, 10] = True
        result, values = assembled_masks(a, b, g), capacity(a, b, g)
        for name, mask in result.items():
            iou = int((mask & g).sum())/int((mask | g).sum())
            assert abs(iou-values[name+'_iou']) < 1e-12
            encoded = encode(mask); encoded['counts'] = encoded['counts'].encode('ascii')
            assert np.array_equal(mu.decode(encoded), mask)
    z = np.zeros((9, 11), dtype=bool); g = z.copy(); g[8, 10] = True
    assert all(not x.any() for x in assembled_masks(z, z, g).values())
    a = z.copy(); a[1, 1] = True; b = a.copy(); b[1, 2] = True
    assert abs(capacity(a, b, g)['whole_iou']) < 1e-12
    assert all(np.array_equal(mask, a) for mask in assembled_masks(a, b, g).values()), 'GT outside crop changed fixed core or stable ties'
    def item(image, category, score, mask):
        return dict(image_id=image, category_id=category, score=score, segmentation=encode(mask),
                    bbox=mu.toBbox(encode(mask)).tolist())
    # Includes exact ties, unmatched, category ties, below-floor and >100 candidates.
    base = [item(1, 1, .8, a), item(1, 2, .8, a), item(1, 3, .001, a)]
    other = [item(1, 1, .8, b), item(1, 4, .8, a)]
    base.extend(item(2, 1, .9-.001*i, a) for i in range(105))
    saved = copy.deepcopy((base, other))
    for kind in ('endpoint', 'union'):
        clean, provenance = templates(base, other, kind)
        reference = [x for x in binary_geometry_control(base, other, kind)[0] if float(x['score']) >= .001]
        assert clean == reference and len(clean) == len(provenance)
        assert Counter(x['image_id'] for x in clean)[2] == 100
    assert (base, other) == saved
    dataset = dict(images=[dict(id=1, height=9, width=11)], categories=[dict(id=1, name='x')],
        annotations=[dict(id=1, image_id=1, category_id=1, segmentation=encode(a),
                          area=1, bbox=mu.toBbox(encode(a)).tolist(), iscrowd=0)])
    pred = [item(1, 1, .9, a)]; before = copy.deepcopy((dataset, pred))
    perfect = full_metrics(dataset, pred)
    assert abs(perfect['AP']-100) < 1e-10 and perfect['tp_by_threshold'] == [1]*10
    assert (dataset, pred) == before
    empty = full_metrics(dataset, [])
    assert empty['AP'] == 0 and empty['tp_by_threshold'] == [0]*10
    assert (dataset, pred) == before
    print(json.dumps(dict(status='passed', random_mask_reconstructions=64, empty_outside_gt_cases=2,
        exact_legacy_controls=2, native_coco_fixtures=2, no_input_mutation=True)), flush=True)


def probe(folder):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    from offline_tools.fuse_coco_instances import rle_for_compute
    from offline_tools.mva_fusion_controls import binary_geometry_control
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for path, h in INPUT_HASHES.items(): assert sha(ROOT/path) == h
    dataset = json.loads((ROOT/GT).read_text()); coco = COCO(str(ROOT/GT)); targets = defaultdict(list)
    for ann in dataset['annotations']:
        if not ann.get('iscrowd', 0): targets[(ann['image_id'], ann['category_id'])].append(ann)
    report = dict(status='running', exploratory=True, gpu_disabled=True, test_split_access=False,
        scope='GT-assisted simultaneously assembled local-IoU policies; NOT AP upper bounds, deployable method, or confirmation',
        policy='same frozen pair matches/order/scores/floor .001/global top100; edits after pruning; all positive-anchor matched candidates edited',
        anchor='same-class noncrowd GT with highest higher-score endpoint IoU; noGT/zero overlap keep endpoint',
        limitations='shared anchors/duplicate detections, conditional subset; fixed core and union support cannot repair common errors; AP need not follow local hierarchy',
        input_sha256=INPUT_HASHES, pairs={})
    for label, paths in [('mask2former_42_3407', [BASES['42'], BASES['3407']]), ('maskrcnn_42_3407', RCNN)]:
        raw = [json.loads((ROOT/p).read_text()) for p in paths]
        outputs, provenance = templates(raw[0], raw[1], 'endpoint')
        union, union_provenance = templates(raw[0], raw[1], 'union')
        assert len(outputs) == len(union) and len(provenance) == len(union_provenance)
        for kind, values in [('endpoint', outputs), ('union', union)]:
            reference = [x for x in binary_geometry_control(raw[0], raw[1], kind)[0] if float(x['score']) >= .001]
            assert values == reference, 'Legacy control candidate identity changed'
        systems = dict(endpoint=outputs, union=union,
                       **{name: copy.deepcopy(outputs) for name in ('whole', 'component', 'pixel')})
        counts = Counter(); rows = []
        for dtid, (item, sources) in enumerate(zip(outputs, provenance), 1):
            assert all(item[k] == union[dtid-1][k] for k in ('image_id', 'category_id', 'score'))
            if sources is None: counts['unmatched_kept'] += 1; continue
            counts['matched_kept'] += 1
            key = (item['image_id'], item['category_id']); anns = targets[key]
            if not anns: counts['no_same_class_gt_kept'] += 1; continue
            a, b = sources; ar, br = [rle_for_compute(x['segmentation']) for x in sources]
            gt_rles = [coco.annToRLE(x) for x in anns]
            chosen = ar if float(a['score']) >= float(b['score']) else br
            overlaps = mu.iou([chosen], gt_rles, [0]*len(anns))[0]
            index = int(np.argmax(overlaps)); endpoint_iou = float(overlaps[index])
            if endpoint_iou <= 0: counts['zero_overlap_anchor_kept'] += 1; continue
            decoded = mu.decode([ar, br, gt_rles[index]]).astype(bool)
            masks = assembled_masks(decoded[:, :, 0], decoded[:, :, 1], decoded[:, :, 2])
            row = dict(dtid=dtid, image_id=key[0], category_id=key[1], gt_id=anns[index]['id'],
                       score=float(item['score']), endpoint_iou=endpoint_iou)
            truth = decoded[:, :, 2]
            old_mask = mu.decode(rle_for_compute(item['segmentation'])).astype(bool)
            for name, mask in masks.items():
                changed = not np.array_equal(mask, old_mask)
                counts[name+'_changed_pixels'] += int(changed)
                rle = encode(mask); systems[name][dtid-1]['segmentation'] = rle
                systems[name][dtid-1]['bbox'] = mu.toBbox(rle_for_compute(rle)).tolist()
                row[name+'_iou'] = int((mask & truth).sum())/int((mask | truth).sum())
            assert row['whole_iou'] <= row['component_iou']+1e-12 <= row['pixel_iou']+2e-12
            rows.append(row)
            if len(rows) % 250 == 0:
                print('ASSEMBLY_PROGRESS {} {}'.format(label, len(rows)), flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 12*1024**2:
                    raise RuntimeError('Own high-water RSS above12GiB')
        assert counts['unmatched_kept']+counts['matched_kept'] == len(outputs)
        assert counts['matched_kept'] == len(rows)+counts['no_same_class_gt_kept']+counts['zero_overlap_anchor_kept']
        with (folder/(label+'_anchors.csv')).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        result = dict(candidate_count=len(outputs), counts=dict(counts), edited_anchors=len(rows),
                      unique_gt_anchors=len({x['gt_id'] for x in rows}), policies={})
        for name in POLICIES:
            values = systems[name]
            assert len(values) == len(outputs)
            assert all(all(a[k] == b[k] for k in ('image_id', 'category_id', 'score')) for a, b in zip(outputs, values))
            output_path = folder/(label+'_'+name+'.json'); save(values, output_path)
            metrics = full_metrics(dataset, values)
            if name in ('endpoint', 'union'):
                assert abs(metrics['AP']-EXPECTED[label][('endpoint', 'union').index(name)]) < 1e-6, 'Historical AP mismatch'
            result['policies'][name] = dict(metrics, output_sha256=sha(output_path))
            print('FULL_COCO_COMPLETE {} {} AP={:.8f}'.format(label, name, metrics['AP']), flush=True)
            report['pairs'][label] = result; save(report, folder/'report.json')
        for name in POLICIES:
            result['policies'][name]['delta_endpoint_AP'] = result['policies'][name]['AP']-result['policies']['endpoint']['AP']
        result['component_minus_whole_AP'] = result['policies']['component']['AP']-result['policies']['whole']['AP']
        result['pixel_minus_component_AP'] = result['policies']['pixel']['AP']-result['policies']['component']['AP']
        report['pairs'][label] = result; save(report, folder/'report.json')
        for path, h in INPUT_HASHES.items(): assert sha(ROOT/path) == h
    report['status'] = 'complete'; save(report, folder/'report.json')


def fingerprint():
    from offline_tools.mva_region_capacity_probe import fingerprint as previous
    result = previous()
    name = str((ENV/'lib/python3.7/site-packages/pycocotools/cocoeval.py').relative_to(ROOT))
    result[name] = sha(ROOT/name)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true'); p.add_argument('--launch', action='store_true')
    p.add_argument('--worker', action='store_true'); p.add_argument('--output-dir', type=Path)
    args = p.parse_args(); os.chdir(str(ROOT))
    if args.self_test: self_test(); return
    folder = args.output_dir.resolve(); folder.relative_to(ROOT/'output_offline')
    if args.launch:
        if folder.exists(): raise RuntimeError('New output directory required')
        manifest = dict(source_sha256={f: sha(ROOT/f) for f in SOURCES}, runtime_sha256=fingerprint(),
                        input_sha256=INPUT_HASHES, gpu_disabled=True, test_split_access=False)
        cpus = sorted(os.sched_getaffinity(0))[:2]; manifest['cpu_affinity'] = cpus
        env = dict(os.environ); env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            LD_LIBRARY_PATH=str(ENV/'lib')+':'+str(ENV/'lib/python3.7/site-packages/torch/lib'))
        folder.mkdir(parents=True); save(manifest, folder/'deployment.json')
        cmd = ['ionice', '-c', '2', '-n', '7', 'nice', '-n', '10', 'taskset', '-c', ','.join(map(str, cpus)),
               str(ENV/'bin/python'), '-u', __file__, '--worker', '--output-dir', str(folder)]
        with (folder/'console.log').open('x') as log:
            child = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        manifest['pid'] = child.pid; save(manifest, folder/'deployment.json'); print(json.dumps(manifest)); return
    if not args.worker: raise RuntimeError('Use --self-test or --launch')
    code, error = 0, None
    try:
        manifest = json.loads((folder/'deployment.json').read_text())
        assert Path(sys.prefix).resolve() == ENV.resolve()
        assert all(sha(ROOT/f) == h for f, h in manifest['source_sha256'].items())
        assert fingerprint() == manifest['runtime_sha256']; probe(folder)
        assert all(sha(ROOT/f) == h for f, h in manifest['source_sha256'].items()), 'Source changed during run'
        assert fingerprint() == manifest['runtime_sha256'], 'Runtime changed during run'
    except BaseException as exc:
        code, error = 1, repr(exc); raise
    finally:
        save(dict(exit_code=code, error=error, finished_unix=time.time()), folder/'completion.json')


if __name__ == '__main__': main()
