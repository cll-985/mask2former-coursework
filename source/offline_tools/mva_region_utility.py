"""Same-budget OOF test of signed, endpoint-relative single-edit utility."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.mva_geometry_probe import ENV, GT, save, sha
from offline_tools.mva_region_residual import OLD_REMOTE, PAIR_INPUTS, arrays, old_cache, payload, read, verify

SPEC = '.aris/compute/mva-region-utility-20261104.md'
SOURCES = ('offline_tools/mva_region_utility.py', 'offline_tools/mva_region_utility_targets.py',
           'offline_tools/mva_region_utility_fit.py', SPEC)
FIT = ROOT/'output_offline/MVA_region_residual_20261104_fit_staged'
AP = ROOT/'output_offline/MVA_region_residual_20261104_v1'
PINS = ((FIT, 'c93eb18884f1f52f503742c8b073ecc3d7142d5f08c4b2453bb4981816494cbb'),
        (AP, '50f346470c24d0d983fbcc0b5ab860fc320d7cced8694eec57ec9b57676e39a5'))


def load_residual():
    for folder, digest in PINS:
        verify(folder, {'output_hashes.json': digest})
        verify(folder, read(folder/'output_hashes.json')['sha256'])
        assert read(folder/'completion.json')['exit_code'] == 0
    assert read(FIT/'fit_report.json')['status'] == 'complete'
    ap = read(AP/'report.json')
    assert ap['status'] == 'complete'
    verify(ROOT, ap['source_sha256'])
    return ap


def collect_targets(coco, data, outputs, provenance, folds):
    import numpy as np
    from pycocotools import mask as mu
    from offline_tools.mva_region_binary_crossfit import decoded_features, guard
    from offline_tools.mva_region_utility_targets import single_flip_utility
    targets = np.full(len(data['ry']), np.nan)
    anchored = np.zeros(len(targets), dtype=bool)
    candidates = data['candidates']
    assert [row['dtid'] for row in candidates] == [i for i, pair in enumerate(provenance, 1) if pair is not None]
    offset = 0
    for index, row in enumerate(candidates, 1):
        dtid = row['dtid']
        item = outputs[dtid-1]
        assert (row['image_id'], row['category_id'], row['fold']) == (
            item['image_id'], item['category_id'], folds[row['image_id']])
        start, stop = row['start'], row['stop']
        assert start == offset and stop-start == row['component_count']
        offset = stop
        gt_id = row['anchor_gt_id']
        if gt_id is not None:
            ann = coco.anns[gt_id]
            assert (ann['image_id'], ann['category_id']) == (row['image_id'], row['category_id'])
            assert not ann.get('iscrowd', 0)
        truth_rle = coco.annToRLE(coco.anns[gt_id]) if gt_id is not None else None
        row['truth_rle'] = truth_rle
        if stop > start:
            sl = slice(start, stop)
            assert np.all(data['rc'][sl] == dtid) and np.all(data['ri'][sl] == row['image_id'])
            if truth_rle is None:
                targets[sl] = 0.0
            else:
                a, b = provenance[dtid-1]
                d = decoded_features(a, b, item['category_id'])
                assert np.array_equal(d['region_features'], data['rx'][sl])
                assert np.array_equal(d['areas'], data['ra'][sl])
                targets[sl] = single_flip_utility(d, mu.decode(truth_rle))
                anchored[sl] = True
        if index % 500 == 0:
            guard()
            print('TARGET_PROGRESS candidates={} components={}'.format(index, offset), flush=True)
    assert offset == len(targets) and np.isfinite(targets).all()
    assert np.count_nonzero(targets[~anchored]) == 0
    return targets, anchored


def edit_summary(data, target, anchored, selected):
    import numpy as np
    chosen = target[selected]
    changed = [float(data['rx'][row['start']:row['stop'], 11][selected[row['start']:row['stop']]].sum())
               for row in data['candidates']]
    return dict(edits=int(selected.sum()), positive=int((chosen > 0).sum()), zero=int((chosen == 0).sum()),
                negative=int((chosen < 0).sum()), no_anchor=int((selected & ~anchored).sum()),
                sum_single_edit_target=float(chosen.sum()),
                mean_single_edit_target=float(chosen.mean()) if chosen.size else None,
                masks_touched=int(np.count_nonzero(changed)), mean_union_fraction_changed=float(np.mean(changed)))


def run(folder):
    import numpy as np
    from pycocotools.coco import COCO
    from offline_tools.mva_region_ap_probe import full_metrics, templates
    from offline_tools.mva_region_binary_crossfit import assemble, fingerprint, guard
    from offline_tools.mva_region_residual_fit import default_bits
    from offline_tools.mva_region_utility_fit import fit_oof
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path(sys.prefix).resolve() == ENV.resolve()
    assert all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    source = {name: sha(ROOT/name) for name in SOURCES}
    pins, deploy, folds = old_cache(OLD_REMOTE)
    residual_report = load_residual()
    assert deploy['runtime_sha256'] == fingerprint()
    verify(ROOT, deploy['input_sha256'])
    dataset = read(ROOT/GT)
    coco = COCO(str(ROOT/GT))
    ids = sorted(int(x['id']) for x in dataset['images'])
    assert set(ids) == set(folds)
    old = read(OLD_REMOTE/'report.json')['pairs']
    report = dict(status='running', exploratory=True, source_sha256=source,
        input_sha256=deploy['input_sha256'], runtime_sha256=deploy['runtime_sha256'],
        cache_sha256=pins, residual_manifest_sha256={str(folder.relative_to(ROOT)): digest for folder, digest in PINS},
        pairs={})
    for label, paths in PAIR_INPUTS.items():
        print('COLLECT '+label, flush=True)
        outputs, provenance = templates(*[read(ROOT/path) for path in paths], 'endpoint')
        data = arrays(OLD_REMOTE/(label+'_features.npz'))
        previous = arrays(OLD_REMOTE/(label+'_oof.npz'))
        residual = arrays(FIT/(label+'_residual_oof.npz'))
        base = default_bits(data['rx'])
        residual_p = residual['flip_probability']
        assert np.array_equal(residual['base_bits'], base)
        assert np.isfinite(residual_p).all() and ((residual_p >= 0) & (residual_p <= 1)).all()
        assert np.array_equal(residual['keep_bits'], base ^ (residual_p > .5))
        data['candidates'] = read(OLD_REMOTE/(label+'_candidate_manifest.json'))
        target, anchored = collect_targets(coco, data, outputs, provenance, folds)
        np.savez_compressed(str(folder/(label+'_utility_targets.npz')), utility_target=target, anchored=anchored)
        predictions, flips, fits = fit_oof(data, folds, target, residual_p, folder, label)
        assert predictions.shape == target.shape and np.isfinite(predictions).all() and flips.dtype == np.dtype(bool)
        row_fold = np.asarray([folds[int(i)] for i in data['ri']])
        oracle = np.zeros(len(flips), dtype=bool)
        for fold in range(5):
            held = np.flatnonzero(row_fold == fold)
            k = int((residual_p[held] > .5).sum())
            assert int(flips[held].sum()) == k
            assert np.array_equal(np.sort(held[np.lexsort((held, -predictions[held]))[:k]]), held[flips[held]])
            oracle[held[np.lexsort((held, -target[held]))[:k]]] = True
        edits = {name: edit_summary(data, target, anchored, mask)
                 for name, mask in (('residual', residual_p > .5), ('utility', flips), ('target_rank_oracle', oracle))}
        assert len({x['edits'] for x in edits.values()}) == 1
        print('ASSEMBLE '+label, flush=True)
        systems, rows, local = assemble(outputs, provenance, data, (base ^ flips).astype(float),
            previous['whole_probability'], folds, ids)
        assert all(systems['region'][i] == outputs[i] for i, pair in enumerate(provenance) if pair is None)
        assert sum(row['region_changed_mask'] for row in rows) == edits['utility']['masks_touched']
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=str(folder)) as tmp:
            whole = Path(tmp)/'whole.json'
            save(systems.pop('whole'), whole)
            assert sha(whole) == old[label]['models']['whole']['output_sha256']
        path = folder/(label+'_utility_predictions.json')
        save(systems['region'], path)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (folder/(label+'_decisions.csv')).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        metrics = full_metrics(dataset, systems['region'])
        ap = metrics['AP']
        delta = dict(residual=ap-residual_report['pairs'][label]['metrics']['AP'],
            v1_region=ap-old[label]['models']['region']['AP'], endpoint=ap-old[label]['endpoint']['AP'],
            learned_whole=ap-old[label]['models']['whole']['AP'],
            strong_baseline=ap-(56.079893043726535 if label == 'mask2former_42_3407' else 53.288416))
        positive_folds = sum(row['region_minus_whole_image_equal_iou'] > 0 for row in local)
        report['pairs'][label] = dict(metrics=metrics, delta_AP=delta, fits=fits, edit_diagnostics=edits,
            local_by_fold=local, positive_local_folds=positive_folds, output_sha256=sha(path),
            pair_gate=delta['learned_whole'] >= .20 and delta['endpoint'] > 0 and positive_folds >= 4)
        save(report, folder/'report.json')
        print('COCO_COMPLETE {} AP={:.8f} {}'.format(label, ap, delta), flush=True)
        guard()
    report['directional_gate_passed'] = all(x['delta_AP']['residual'] > 0 for x in report['pairs'].values())
    report['advancement_gate_passed'] = (report['directional_gate_passed']
        and all(x['pair_gate'] for x in report['pairs'].values())
        and report['pairs']['mask2former_42_3407']['delta_AP']['strong_baseline'] > 0)
    verify(ROOT, source)
    verify(ROOT, deploy['input_sha256'])
    verify(OLD_REMOTE, pins)
    load_residual()
    assert deploy['runtime_sha256'] == fingerprint()
    report['status'] = 'complete'
    save(report, folder/'report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.self_test:
        from offline_tools import mva_region_utility_fit as fit, mva_region_utility_targets as target
        print(json.dumps(dict(fit_witness=fit.self_test(), target_witness=target.self_test())), flush=True)
        if args.native:
            from offline_tools.mva_region_binary_crossfit import self_test
            self_test()
        return
    if args.output_dir is None:
        parser.error('Use --self-test or set --output-dir')
    folder = args.output_dir.resolve()
    folder.relative_to(ROOT/'output_offline')
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    code, error = 0, None
    try:
        run(folder)
    except BaseException as exc:
        code, error = 1, repr(exc)
        raise
    finally:
        save(dict(exit_code=code, error=error, elapsed_seconds=time.monotonic()-start), folder/'completion.json')
        save(dict(sha256=payload(folder)), folder/'output_hashes.json')


if __name__ == '__main__':
    main()
