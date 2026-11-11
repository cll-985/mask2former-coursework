"""GT-assisted local capacity oracles; NOT a method, AP bound, or confirmation."""
import argparse
from collections import defaultdict
import csv
from functools import cmp_to_key
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.mva_geometry_probe import ENV, GT, EXPECTED_GT, BASES, RCNN, sha, save

SPEC = '.aris/compute/mva-region-capacity-probe-20261102.md'
SOURCES = ['offline_tools/mva_region_capacity_probe.py', 'offline_tools/mva_geometry_probe.py',
           'offline_tools/compare_fusion_rules.py', 'offline_tools/fuse_coco_instances.py', SPEC,
           '.aris/compute/mva-geometry-probe-20261101.md', '.aris/compute/mva-arch-analysis-20261030.md']
INPUT_HASHES = {
    GT: EXPECTED_GT,
    BASES['42']: 'bee0182e150ca5a02132993ffb7ced95100475ce5a9b355ab6a613e8d435b7a0',
    BASES['3407']: '33c6eb1a73dfdfea7d79f1d13778b1b7dd2f66d9487989d72b6b4e9b0471577c',
    RCNN[0]: '645edf384c65b30871a8bf1eeba4959aa7e4fe8ab7d3cf6328e83621fc4ec977',
    RCNN[1]: '12001a6dad31a60d0a6e6e4fa9e9f25dfca7f48faa520916503a96a7e2c6c396'}


def best_component_prefix(t0, d0, components):
    """Exact optimal fraction over any subset of disjoint membership components.

    Sort by t/(t+f), equivalent to t/f including f=0. The optimal subset
    includes regions satisfying t-lambda*f>0 at lambda=J*. Some sorted prefix
    therefore attains J*. Integer cross-products fix sorting and best ties.
    This is a classical fractional-selection oracle, not a novel algorithm.
    """
    assert d0 > 0 and 0 <= t0 <= d0
    assert all(t >= 0 and f >= 0 and t+f > 0 for t, f in components)
    def compare(i, j):
        t, f = components[i]; u, v = components[j]
        cross = t*(u+v)-u*(t+f)
        return -1 if cross > 0 else (1 if cross < 0 else 0)
    order = sorted(range(len(components)), key=cmp_to_key(compare))
    numerator, denominator = best_num, best_den = int(t0), int(d0)
    best_count = 0
    for count, index in enumerate(order, 1):
        t, f = components[index]; numerator += t; denominator += f
        if numerator*best_den > best_num*denominator:
            best_num, best_den, best_count = numerator, denominator, count
    return best_num, best_den, order[:best_count]


def membership_components(a, b, truth):
    import numpy as np
    from scipy import ndimage
    stats, labels_by_side = [], []
    for region in (a & ~b, b & ~a):
        labels, count = ndimage.label(np.ascontiguousarray(region), structure=np.ones((3, 3), dtype=np.uint8))
        labels_by_side.append(labels)
        areas = np.bincount(labels.ravel(), minlength=count+1)
        true_areas = np.bincount(labels[truth].ravel(), minlength=count+1)
        stats.extend((int(true_areas[i]), int(areas[i]-true_areas[i])) for i in range(1, count+1))
    return stats, labels_by_side


def capacity(a, b, truth, crop=True):
    import numpy as np
    a, b, truth = [np.asarray(x, dtype=bool) for x in (a, b, truth)]
    assert a.shape == b.shape == truth.shape and a.ndim == 2
    g = int(truth.sum()); assert g > 0
    union = a | b
    if crop and union.any():
        yy, xx = np.nonzero(union)
        box = np.s_[int(yy.min()):int(yy.max())+1, int(xx.min()):int(xx.max())+1]
        a, b, truth = [x[box] for x in (a, b, truth)]
    elif crop:
        # Avoid labelling a full-resolution all-zero field; GT area remains g.
        a, b, truth = [x[:1, :1] & False for x in (a, b, truth)]
    core, union = a & b, a | b
    t0, f0 = int((core & truth).sum()), int((core & ~truth).sum())
    components, _ = membership_components(a, b, truth)
    menu = [(int((m & truth).sum()), g+int((m & ~truth).sum())) for m in (a, b, union, core)]
    whole_index = 0
    for i in range(1, 4):
        if menu[i][0]*menu[whole_index][1] > menu[whole_index][0]*menu[i][1]: whole_index = i
    wn, wd = menu[whole_index]
    cn, cd, selected = best_component_prefix(t0, g+f0, components)
    pn, pd = int((union & truth).sum()), g+f0
    assert wn*cd <= cn*wd and cn*pd <= pn*cd, 'Oracle hierarchy broken'
    assert sum(t+f for t, f in components) == int((a ^ b).sum())
    whole, component, pixel = wn/wd, cn/cd, pn/pd
    return dict(whole_iou=whole, component_iou=component, pixel_iou=pixel,
        whole_menu_index=whole_index, component_count=len(components), selected_components=len(selected),
        mixed_pixel_components=sum(t > 0 and f > 0 for t, f in components),
        pure_true_components=sum(t > 0 and f == 0 for t, f in components),
        pure_false_components=sum(t == 0 and f > 0 for t, f in components),
        disagreement_tp=sum(t for t, f in components), disagreement_fp=sum(f for t, f in components),
        common_fn=g-pn, common_fp=f0, common_fn_fraction_gt=(g-pn)/g, common_fp_fraction_gt=f0/g,
        component_minus_whole=component-whole, pixel_minus_component=pixel-component,
        pixel_minus_whole=pixel-whole, component_gt_whole=int(cn*wd > wn*cd),
        pixel_gt_component=int(pn*cd > cn*pd), gt_area=g)


def summarize(rows):
    fields = ('whole_iou', 'component_iou', 'pixel_iou', 'component_minus_whole',
              'pixel_minus_component', 'pixel_minus_whole', 'common_fn_fraction_gt', 'common_fp_fraction_gt')
    result = dict(n=len(rows), image_count=len({r['image_id'] for r in rows}),
                  component_gt_whole=sum(r['component_gt_whole'] for r in rows),
                  pixel_gt_component=sum(r['pixel_gt_component'] for r in rows))
    by_image = defaultdict(list)
    for row in rows: by_image[row['image_id']].append(row)
    result['matched_row_means'] = {k: sum(r[k] for r in rows)/len(rows) if rows else None for k in fields}
    result['image_equal_means'] = {k: sum(sum(r[k] for r in rs)/len(rs) for rs in by_image.values())/len(by_image)
                                  if by_image else None for k in fields}
    return result


def self_test():
    import numpy as np
    from scipy import ndimage
    from itertools import product
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert Path(sys.prefix).resolve() == ENV.resolve(), 'Use frozen native runtime'
    def brute(t0, d0, stats):
        best = (t0, d0)
        for bits in product((0, 1), repeat=len(stats)):
            n = t0+sum(t*z for (t, f), z in zip(stats, bits))
            d = d0+sum(f*z for (t, f), z in zip(stats, bits))
            if n*best[1] > best[0]*d: best = n, d
        return best
    exhaustive = 0
    choices = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]
    for stats in product(choices, repeat=3):
        for t0, d0 in ((0, 1), (1, 1), (1, 3)):
            n, d, _ = best_component_prefix(t0, d0, stats); bn, bd = brute(t0, d0, stats)
            assert n*bd == bn*d; exhaustive += 1
    rng = np.random.RandomState(20261102)
    for _ in range(256):
        count = int(rng.randint(0, 8)); d0 = int(rng.randint(1, 15)); t0 = int(rng.randint(0, d0+1))
        stats = [tuple(map(int, rng.randint(0, 8, size=2))) for i in range(count)]
        stats = [(t, f) if t+f else (0, 1) for t, f in stats]
        n, d, _ = best_component_prefix(t0, d0, stats); bn, bd = brute(t0, d0, stats)
        assert n*bd == bn*d
    assert best_component_prefix(1, 2, [(1, 2), (1, 2)])[2] == []
    def component_sets(labels):
        return {frozenset(map(tuple, np.argwhere(labels == i))) for i in range(1, int(labels.max())+1)}
    def flood(mask):
        unvisited = set(map(tuple, np.argwhere(mask))); answer = set()
        while unvisited:
            todo = [unvisited.pop()]; group = set(todo)
            while todo:
                y, x = todo.pop()
                for dy, dx in product((-1, 0, 1), repeat=2):
                    q = y+dy, x+dx
                    if q in unvisited: unvisited.remove(q); group.add(q); todo.append(q)
            answer.add(frozenset(group))
        return answer
    for _ in range(200):
        mask = rng.rand(int(rng.randint(1, 13)), int(rng.randint(1, 13))) < .3
        labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        assert component_sets(labels) == flood(mask)
    def make(points):
        m = np.zeros((10, 10), dtype=bool)
        for y, x in points: m[y, x] = True
        return m
    empty = make([]); g = make([(0, 0), (9, 9)])
    fixtures = [(empty, empty, g),
        (make([(0, 0)]), g, g),
        (make([(1, 1)]), make([(2, 2)]), g),
        (make([(0, 0)]), make([(0, 0)]), g),
        (make([(0, 0), (1, 1)]), make([(0, 0), (1, 1), (9, 9)]), g),
        (make([(1, 1)]), make([(1, 2)]), make([(1, 1)]))]
    core_points = [(4, x) for x in range(1, 9)]
    ga = make(core_points+[(1, 1), (7, 7)])
    aa = make(core_points+[(1, 1), (1, 8)])
    bb = make(core_points+[(7, 7), (8, 1)])
    fixtures.append((aa, bb, ga))
    for a, b, truth in fixtures:
        r, full = capacity(a, b, truth), capacity(a, b, truth, crop=False)
        assert r == full, 'Cropping changed oracle, including GT outside crop'
        core = a & b; _, labels = membership_components(a, b, truth)
        assert np.array_equal(core | (labels[0] > 0), a)
        assert np.array_equal(core | (labels[1] > 0), b)
        assert np.array_equal(core | (labels[0] > 0) | (labels[1] > 0), a | b)
        assert np.array_equal(core, a & b)
    assert capacity(aa, bb, ga)['component_gt_whole'] == 1
    assert membership_components(fixtures[-2][0], fixtures[-2][1], fixtures[-2][2])[0] == [(1, 0), (0, 1)]
    # Local maximum IoU need not preserve two-instance greedy TP count.
    def flat(indices): return make([(i//10, i % 10) for i in indices])
    g1, g2 = flat(range(3)), flat(range(3, 9))
    a, c = flat(range(6)), flat(list(range(2))+list(range(3, 9)))
    def iou(m, t): return int((m & t).sum())/int((m | t).sum())
    def greedy_count(predictions):
        available = set((0, 1)); count = 0
        for prediction in predictions:
            candidates = [(iou(prediction, target), i) for i, target in enumerate((g1, g2)) if i in available]
            if candidates:
                value, index = max(candidates)
                if value >= .5: available.remove(index); count += 1
        return count
    assert iou(c, g2) > iou(a, g2) and greedy_count([a, g2]) == 2 and greedy_count([c, g2]) == 1
    print(json.dumps(dict(status='passed', fractional_exhaustive_cases=exhaustive, fractional_random_cases=256,
        seeded_ccl_masks=200, array_fixtures=len(fixtures), local_assignment_warning_fixture=True,
        scope='synthetic oracle arithmetic/CCL only; no native COCO AP or real inputs')), flush=True)


def probe(folder):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    from offline_tools.compare_fusion_rules import group_results
    from offline_tools.fuse_coco_instances import greedy_matches, rle_for_compute
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for path, expected in INPUT_HASHES.items(): assert sha(ROOT/path) == expected, 'Frozen input changed: '+path
    coco = COCO(str(ROOT/GT)); gt_groups = defaultdict(list)
    for annotation in coco.dataset['annotations']:
        if not annotation.get('iscrowd', 0): gt_groups[(annotation['image_id'], annotation['category_id'])].append(annotation)
    report = dict(status='running', exploratory=True, gpu_disabled=True, test_split_access=False,
        scope='GT-assisted local capacity only; not AP bound, deployable gain, or independent confirmation',
        selection='greedy maskIoU>=.5; mean score>=.001 BEFORE final global top100; positive GT anchor required',
        anchor='same-class noncrowd GT maximizing higher-score endpoint IoU; selected with GT, biased subset',
        components='A\\B and B\\A separately 8-connected; fixed core A intersect B; no component budget',
        input_sha256=INPUT_HASHES, pairs={})
    for label, paths in [('mask2former_42_3407', [BASES['42'], BASES['3407']]), ('maskrcnn_42_3407', RCNN)]:
        raw = [json.loads((ROOT/p).read_text()) for p in paths]; groups = [group_results(x) for x in raw]
        counts = dict(matched_before_floor=0, below_mean_score_floor=0, no_same_class_gt=0,
                      zero_overlap_anchor=0, empty_union=0); rows = []
        with (folder/(label+'.csv')).open('x', newline='') as f:
            writer = None
            for key in sorted(set(groups[0]) | set(groups[1])):
                left, right = [sorted(group[key], key=lambda x: -float(x['score'])) for group in groups]
                annotations = gt_groups[key]; targets = [coco.annToRLE(x) for x in annotations]
                for i, j, _ in greedy_matches(left, right, .5):
                    counts['matched_before_floor'] += 1; a, b = left[i], right[j]
                    score = (float(a['score'])+float(b['score']))/2
                    if score < .001: counts['below_mean_score_floor'] += 1; continue
                    if not annotations: counts['no_same_class_gt'] += 1; continue
                    ar, br = [rle_for_compute(x['segmentation']) for x in (a, b)]
                    overlaps = mu.iou([ar, br], targets, [0]*len(targets))
                    chosen = 0 if float(a['score']) >= float(b['score']) else 1
                    target_index = int(np.argmax(overlaps[chosen])); endpoint_iou = float(overlaps[chosen, target_index])
                    if endpoint_iou <= 0: counts['zero_overlap_anchor'] += 1; continue
                    decoded = mu.decode([ar, br, targets[target_index]]).astype(bool)
                    if not (decoded[:, :, 0] | decoded[:, :, 1]).any(): counts['empty_union'] += 1; continue
                    row = dict(image_id=key[0], category_id=key[1], gt_id=annotations[target_index]['id'],
                        score=score, endpoint_iou=endpoint_iou, endpoint_ge05=int(endpoint_iou >= .5),
                        **capacity(decoded[:, :, 0], decoded[:, :, 1], decoded[:, :, 2]))
                    assert row['gt_area'] == int(mu.area(targets[target_index]))
                    if writer is None: writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row); rows.append(row)
                    if len(rows) % 250 == 0:
                        f.flush(); print('CAPACITY_PROGRESS {} {}'.format(label, len(rows)), flush=True)
                        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 12*1024**2:
                            raise RuntimeError('Own high-water RSS above12GiB')
        assert counts['matched_before_floor'] == len(rows)+sum(v for k, v in counts.items() if k != 'matched_before_floor')
        for p in paths: assert sha(ROOT/p) == INPUT_HASHES[p]
        result = dict(counts, diagnosed=len(rows), all_anchored=summarize(rows),
                      endpoint_ge05=summarize([r for r in rows if r['endpoint_ge05']]))
        report['pairs'][label] = result; save(report, folder/'report.json')
        print('PAIR_COMPLETE '+label+' '+json.dumps(result), flush=True)
    for path, expected in INPUT_HASHES.items(): assert sha(ROOT/path) == expected
    report['status'] = 'complete'; save(report, folder/'report.json')


def fingerprint():
    paths = [ENV/'bin/python']+[ENV/('lib/python3.7/site-packages/'+p) for p in (
        'pycocotools/coco.py', 'pycocotools/mask.py', 'scipy/__init__.py', 'scipy/ndimage/__init__.py',
        'scipy/ndimage/measurements.py', 'scipy/ndimage/_ni_support.py')]
    for pattern in ('numpy/core/_multiarray_umath*.so', 'pycocotools/_mask*.so',
                    'scipy/ndimage/_nd_image*.so', 'scipy/ndimage/_ni_label*.so'):
        matches = list(ENV.glob('lib/python3.7/site-packages/'+pattern))
        assert len(matches) == 1, 'Ambiguous native binary: '+pattern; paths.extend(matches)
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
        sources, runtime = {f: sha(ROOT/f) for f in SOURCES}, fingerprint()
        folder.mkdir(parents=True); cpus = sorted(os.sched_getaffinity(0))[:2]; env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            LD_LIBRARY_PATH=str(ENV/'lib')+':'+str(ENV/'lib/python3.7/site-packages/torch/lib'))
        manifest = dict(source_sha256=sources, runtime_sha256=runtime, input_sha256=INPUT_HASHES,
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
        assert fingerprint() == manifest['runtime_sha256']; probe(folder)
    except BaseException as exc:
        code, error = 1, repr(exc); raise
    finally:
        save(dict(exit_code=code, error=error, finished_unix=time.time()), folder/'completion.json')


if __name__ == '__main__': main()
