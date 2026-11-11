"""Frozen v1 keep-bit versus endpoint-relative edit crossfit."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.mva_geometry_probe import BASES, ENV, GT, RCNN, save, sha

SPEC = '.aris/compute/mva-region-residual-20261104.md'
SOURCES = ('offline_tools/mva_region_residual.py', 'offline_tools/mva_region_residual_fit.py', SPEC)
OLD_REMOTE = ROOT/'output_offline/MVA_region_binary_crossfit_20261103_v1'
OLD_LOCAL = ROOT/'artifacts/mva_region_binary_crossfit_20261103'
OLD_HASH = '9d1a6051c4cbfcf50bad126f6cb84bce604b1cbbc8ff0e974665a5c7001cb841'
PAIR_INPUTS = {'mask2former_42_3407': [BASES['42'], BASES['3407']], 'maskrcnn_42_3407': RCNN}
PAIR_FILES = ('_features.npz', '_oof.npz', '_candidate_manifest.json', '_decisions.csv')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(folder, hashes):
    for name, expected in hashes.items():
        if sha(folder/name) != expected:
            raise RuntimeError('SHA-256 mismatch: '+str(folder/name))


def old_cache(folder):
    names = ['output_hashes.json', 'deployment.json', 'report.json', 'fold_manifest.json']
    names += [label+suffix for label in PAIR_INPUTS for suffix in PAIR_FILES]
    hashes = read(folder/'output_hashes.json')['sha256']
    pins = {name: hashes[name] for name in names[1:]}
    pins[names[0]] = OLD_HASH
    verify(folder, pins)
    deploy = read(folder/'deployment.json')
    verify(ROOT, deploy['source_sha256'])
    folds = {int(key): value for key, value in read(folder/'fold_manifest.json').items()}
    assert len(folds) == 810 and sorted(list(folds.values())).count(0) == 162
    return pins, deploy, folds


def arrays(path):
    import numpy as np
    with np.load(str(path), allow_pickle=False) as values:
        return dict(values)


def diagnostic(data, candidates, base, direct, residual):
    import numpy as np
    from offline_tools.mva_region_binary_fit import weights
    weight = weights(data['ri'], data['rc'], data['ra'])
    positive = np.zeros(len(base), dtype=bool)
    change = {key: {'anchored': [], 'no_anchor': []} for key in ('v1_direct', 'residual')}
    for row in candidates:
        start, stop = row['start'], row['stop']
        sl = slice(start, stop)
        assert stop-start == row['component_count']
        assert np.all(data['rc'][sl] == row['dtid']) and np.all(data['ri'][sl] == row['image_id'])
        anchored = row['anchor_gt_id'] is not None
        positive[sl] = anchored
        for key, bits in (('v1_direct', direct), ('residual', residual)):
            changed = bits[sl] != base[sl]
            change[key]['anchored' if anchored else 'no_anchor'].append(
                (bool(changed.any()), float(data['rx'][sl, 11][changed].sum())))
    assert np.array_equal(base[~positive], data['ry'][~positive])
    agreement = {}
    for key, bits in (('endpoint', base), ('v1_direct', direct), ('residual', residual)):
        agreement[key] = {}
        for group, idx in (('anchored', positive), ('no_anchor', ~positive), ('all', np.ones_like(positive))):
            agreement[key][group] = float(np.average(bits[idx] == data['ry'][idx], weights=weight[idx]))
    result = {}
    for name, groups in change.items():
        result[name] = {}
        for group, rows in groups.items():
            values = np.asarray(rows, dtype=float)
            result[name][group] = dict(candidates=len(rows), edited_candidates=int(values[:, 0].sum()),
                changed_fraction=float(values[:, 0].mean()), mean_changed_union_fraction=float(values[:, 1].mean()))
    return dict(weighted_teacher_agreement=agreement, edits=result)


def payload(folder):
    return {path.name: sha(path) for path in sorted(folder.iterdir()) if path.is_file()
            and path.name not in ('output_hashes.json', 'console.log') and not path.name.endswith('.tmp')}


def fit(folder):
    import numpy as np
    import scipy
    from offline_tools.mva_region_residual_fit import fit_oof
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1')
    assert all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    pins, _, folds = old_cache(OLD_LOCAL)
    sources = {name: sha(ROOT/name) for name in SOURCES}
    report = dict(status='running', environment=dict(python=sys.version, platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__), input_sha256=pins, source_sha256=sources, pairs={})
    for label in PAIR_INPUTS:
        data = arrays(OLD_LOCAL/(label+'_features.npz'))
        candidates = read(OLD_LOCAL/(label+'_candidate_manifest.json'))
        probability = arrays(OLD_LOCAL/(label+'_oof.npz'))['region_probability']
        flip, base, bits, logs = fit_oof(data, folds, folder, label)
        assert len(flip) == len(base) == len(probability) and len(logs) == 5
        direct = np.where(probability == .5, base, probability > .5)
        report['pairs'][label] = diagnostic(data, candidates, base, direct, bits)
        save(report, folder/'fit_report.json')
    verify(OLD_LOCAL, pins)
    verify(ROOT, sources)
    report['status'] = 'complete'
    save(report, folder/'fit_report.json')


def evaluate(folder, fitted):
    import numpy as np
    from pycocotools.coco import COCO
    from offline_tools.mva_region_ap_probe import full_metrics, templates
    from offline_tools.mva_region_binary_crossfit import assemble, fingerprint, guard
    from offline_tools.mva_region_residual_fit import default_bits
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path(sys.prefix).resolve() == ENV.resolve()
    pins, deploy, folds = old_cache(OLD_REMOTE)
    manifest = read(fitted/'output_hashes.json')['sha256']
    verify(fitted, manifest)
    fit_report = read(fitted/'fit_report.json')
    assert fit_report['status'] == 'complete' and fit_report['input_sha256'] == pins
    verify(ROOT, fit_report['source_sha256'])
    assert deploy['runtime_sha256'] == fingerprint()
    verify(ROOT, deploy['input_sha256'])
    dataset = read(ROOT/GT)
    coco = COCO(str(ROOT/GT))
    image_ids = sorted(int(item['id']) for item in dataset['images'])
    assert set(image_ids) == set(folds)
    old_report = read(OLD_REMOTE/'report.json')
    report = dict(status='running', exploratory=True, source_sha256=fit_report['source_sha256'],
        input_sha256=deploy['input_sha256'], runtime_sha256=deploy['runtime_sha256'],
        fit_payload_sha256=manifest, old_cache_sha256=pins, pairs={})
    for label, paths in PAIR_INPUTS.items():
        print('ASSEMBLE '+label, flush=True)
        data = arrays(OLD_REMOTE/(label+'_features.npz'))
        previous = arrays(OLD_REMOTE/(label+'_oof.npz'))
        result = arrays(fitted/(label+'_residual_oof.npz'))
        base = default_bits(data['rx'])
        keep = result['keep_bits']
        assert np.array_equal(result['base_bits'], base)
        assert np.isfinite(result['flip_probability']).all()
        assert np.array_equal(keep, base ^ (result['flip_probability'] > .5))
        candidates = read(OLD_REMOTE/(label+'_candidate_manifest.json'))
        for row in candidates:
            gt_id = row['anchor_gt_id']
            row['truth_rle'] = coco.annToRLE(coco.anns[gt_id]) if gt_id is not None else None
        data['candidates'] = candidates
        outputs, provenance = templates(*[read(ROOT/path) for path in paths], 'endpoint')
        assert [row['dtid'] for row in candidates] == [i for i, pair in enumerate(provenance, 1) if pair is not None]
        assert all((row['image_id'], row['category_id'], row['fold']) ==
            (outputs[row['dtid']-1]['image_id'], outputs[row['dtid']-1]['category_id'], folds[row['image_id']])
            for row in candidates)
        systems, rows, local = assemble(outputs, provenance, data, keep.astype(float),
            previous['whole_probability'], folds, image_ids)
        old = old_report['pairs'][label]
        assert len(rows) == old['matched'] and len(outputs) == len(systems['region'])
        assert all(systems['region'][i] == outputs[i] for i, pair in enumerate(provenance) if pair is None)
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=str(folder)) as temp:
            whole_path = Path(temp)/'whole.json'
            save(systems.pop('whole'), whole_path)
            assert sha(whole_path) == old['models']['whole']['output_sha256']
        path = folder/(label+'_residual_predictions.json')
        save(systems['region'], path)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (folder/(label+'_decisions.csv')).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        metrics = full_metrics(dataset, systems['region'])
        diag = fit_report['pairs'][label]
        assert sum(row['region_changed_mask'] for row in rows if row['anchor_gt_id'] is None) == diag['edits']['residual']['no_anchor']['edited_candidates']
        ap = metrics['AP']
        delta = dict(endpoint=ap-old['endpoint']['AP'], v1_direct=ap-old['models']['region']['AP'],
            learned_whole=ap-old['models']['whole']['AP'],
            strong_baseline=ap-(56.079893043726535 if label == 'mask2former_42_3407' else 53.288416))
        positive_folds = sum(row['region_minus_whole_image_equal_iou'] > 0 for row in local)
        report['pairs'][label] = dict(metrics=metrics, delta_AP=delta, local_by_fold=local,
            positive_local_folds=positive_folds, pair_gate=delta['learned_whole'] >= .20 and delta['endpoint'] > 0
            and positive_folds >= 4, diagnostic=diag, output_sha256=sha(path))
        save(report, folder/'report.json')
        print('COCO_COMPLETE {} AP={:.8f} {}'.format(label, ap, delta), flush=True)
        guard()
    report['both_residual_AP_higher_than_v1'] = all(pair['delta_AP']['v1_direct'] > 0 for pair in report['pairs'].values())
    report['old_practical_gate_passed'] = all(pair['pair_gate'] for pair in report['pairs'].values()) and (
        report['pairs']['mask2former_42_3407']['delta_AP']['strong_baseline'] > 0)
    verify(ROOT, fit_report['source_sha256'])
    verify(ROOT, deploy['input_sha256'])
    verify(OLD_REMOTE, pins)
    verify(fitted, manifest)
    assert deploy['runtime_sha256'] == fingerprint()
    report['status'] = 'complete'
    save(report, folder/'report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('self-test', 'fit', 'evaluate'):
        mode.add_argument('--'+flag, action='store_true')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--fit-dir', type=Path)
    args = parser.parse_args()
    if args.self_test:
        from offline_tools.mva_region_residual_fit import self_test
        print(json.dumps(self_test()), flush=True)
        if args.native:
            from offline_tools.mva_region_binary_crossfit import self_test as native_witness
            native_witness()
        return
    if args.output_dir is None or (args.evaluate and args.fit_dir is None):
        parser.error('The selected mode requires an output directory and evaluation requires --fit-dir')
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    code, error = 0, None
    try:
        if args.fit:
            fit(folder)
        else:
            evaluate(folder, args.fit_dir.resolve())
    except BaseException as exc:
        code, error = 1, repr(exc)
        raise
    finally:
        save(dict(exit_code=code, error=error, elapsed_seconds=time.monotonic()-start), folder/'completion.json')
        save(dict(sha256=payload(folder)), folder/'output_hashes.json')


if __name__ == '__main__':
    main()
