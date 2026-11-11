"""Frozen eight-column direction repair of the signed utility diagnostic."""
import argparse
import csv
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.mva_geometry_probe import ENV, GT, save, sha
from offline_tools.mva_region_residual import OLD_REMOTE, PAIR_INPUTS, arrays, old_cache, payload, read, verify
from offline_tools.mva_region_utility import AP, FIT, edit_summary, load_residual

UTILITY = ROOT/'output_offline/MVA_region_utility_20261104_v1'
SOURCES = ('offline_tools/mva_region_direction.py', 'offline_tools/mva_region_direction_fit.py',
           '.aris/compute/mva-region-direction-20261105.md')
UTILITY_MANIFEST = '0d16cd9141b27c85be7b56052cfed27aa0a0cc4caf5cca8d767a2c3bb4d471f6'


def run(folder):
    import numpy as np
    from pycocotools.coco import COCO
    from offline_tools.mva_region_ap_probe import templates, full_metrics
    from offline_tools.mva_region_binary_crossfit import assemble, fingerprint, guard
    from offline_tools.mva_region_residual_fit import default_bits
    from offline_tools.mva_region_direction_fit import fit_oof
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and Path(sys.prefix).resolve() == ENV.resolve()
    assert all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    verify(UTILITY, {'output_hashes.json': UTILITY_MANIFEST})
    utility_pins = read(UTILITY/'output_hashes.json')['sha256']
    verify(UTILITY, utility_pins)
    previous = read(UTILITY/'report.json')
    assert previous['status'] == 'complete' and read(UTILITY/'completion.json')['exit_code'] == 0
    verify(ROOT, previous['source_sha256'])
    residual = load_residual()
    pins, deployment, folds = old_cache(OLD_REMOTE)
    verify(ROOT, deployment['input_sha256'])
    assert deployment['runtime_sha256'] == fingerprint()
    source = {name: sha(ROOT/name) for name in SOURCES}
    dataset = read(ROOT/GT)
    coco = COCO(str(ROOT/GT))
    ids = sorted(int(row['id']) for row in dataset['images'])
    assert set(ids) == set(folds)
    old = read(OLD_REMOTE/'report.json')['pairs']
    report = dict(status='running', exploratory=True, source_sha256=source,
        previous_source_sha256=previous['source_sha256'], input_sha256=deployment['input_sha256'],
        runtime_sha256=deployment['runtime_sha256'], utility_manifest_sha256=UTILITY_MANIFEST,
        old_cache_sha256=pins, pairs={})
    for label, paths in PAIR_INPUTS.items():
        print('FIT '+label, flush=True)
        data = arrays(OLD_REMOTE/(label+'_features.npz'))
        data['candidates'] = read(OLD_REMOTE/(label+'_candidate_manifest.json'))
        target = arrays(UTILITY/(label+'_utility_targets.npz'))
        residual_oof = arrays(FIT/(label+'_residual_oof.npz'))
        base = default_bits(data['rx'])
        assert np.array_equal(base, residual_oof['base_bits'])
        assert np.array_equal(base ^ (residual_oof['flip_probability'] > .5), residual_oof['keep_bits'])
        assert target['utility_target'].shape == base.shape and target['anchored'].shape == base.shape
        prediction, flips, fits = fit_oof(data, folds, target['utility_target'],
            residual_oof['flip_probability'], folder, label)
        row_folds = np.asarray([folds[int(i)] for i in data['ri']])
        for fold in range(5):
            held = np.flatnonzero(row_folds == fold)
            k = int((residual_oof['flip_probability'][held] > .5).sum())
            assert flips[held].sum() == k
            chosen = held[np.lexsort((held, -prediction[held]))[:k]]
            assert np.array_equal(np.sort(chosen), held[flips[held]])
        summary = edit_summary(data, target['utility_target'], target['anchored'], flips)
        assert summary['edits'] == previous['pairs'][label]['edit_diagnostics']['utility']['edits']
        for row in data['candidates']:
            gt_id = row['anchor_gt_id']
            row['truth_rle'] = coco.annToRLE(coco.anns[gt_id]) if gt_id is not None else None
        outputs, provenance = templates(*[read(ROOT/p) for p in paths], 'endpoint')
        assert [row['dtid'] for row in data['candidates']] == [i for i,p in enumerate(provenance,1) if p is not None]
        systems, rows, local = assemble(outputs, provenance, data, (base ^ flips).astype(float),
            arrays(OLD_REMOTE/(label+'_oof.npz'))['whole_probability'], folds, ids)
        assert all(systems['region'][i] == outputs[i] for i,p in enumerate(provenance) if p is None)
        assert sum(row['region_changed_mask'] for row in rows) == summary['masks_touched']
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=str(folder)) as temp:
            whole = Path(temp)/'whole.json'
            save(systems.pop('whole'), whole)
            assert sha(whole) == old[label]['models']['whole']['output_sha256']
        path = folder/(label+'_direction_predictions.json')
        save(systems['region'], path)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (folder/(label+'_decisions.csv')).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        metrics = full_metrics(dataset, systems['region'])
        ap = metrics['AP']
        delta = dict(utility=ap-previous['pairs'][label]['metrics']['AP'],
            residual=ap-residual['pairs'][label]['metrics']['AP'],
            endpoint=ap-old[label]['endpoint']['AP'], learned_whole=ap-old[label]['models']['whole']['AP'],
            strong_baseline=ap-(56.079893043726535 if label == 'mask2former_42_3407' else 53.288416))
        positive_folds = sum(row['region_minus_whole_image_equal_iou'] > 0 for row in local)
        report['pairs'][label] = dict(metrics=metrics, delta_AP=delta, fits=fits, edit_diagnostics=summary,
            local_by_fold=local, positive_local_folds=positive_folds, output_sha256=sha(path),
            pair_gate=delta['endpoint'] > 0 and delta['learned_whole'] >= .20 and positive_folds >= 4)
        save(report, folder/'report.json')
        print('COCO_COMPLETE {} AP={:.8f} {}'.format(label,ap,delta), flush=True)
        guard()
    report['directional_gate_passed'] = all(r['delta_AP']['utility'] > 0 and r['delta_AP']['residual'] > 0
                                           for r in report['pairs'].values())
    report['advancement_gate_passed'] = report['directional_gate_passed'] and all(
        r['pair_gate'] for r in report['pairs'].values()) and report['pairs']['mask2former_42_3407']['delta_AP']['strong_baseline'] > 0
    verify(ROOT, source)
    verify(ROOT, previous['source_sha256'])
    verify(ROOT, deployment['input_sha256'])
    verify(UTILITY, utility_pins)
    verify(OLD_REMOTE,pins)
    load_residual()
    assert deployment['runtime_sha256'] == fingerprint()
    report['status'] = 'complete'
    save(report, folder/'report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.self_test:
        from offline_tools.mva_region_direction_fit import self_test
        import json
        print(json.dumps(self_test()), flush=True)
        if args.native:
            from offline_tools.mva_region_binary_crossfit import self_test as native
            native()
        return
    if args.output_dir is None:
        parser.error('Use --self-test or --output-dir')
    folder = args.output_dir.resolve()
    folder.relative_to(ROOT/'output_offline')
    folder.mkdir(parents=True, exist_ok=False)
    start, code, error = time.monotonic(), 0, None
    try:
        run(folder)
    except BaseException as exc:
        code, error = 1, repr(exc)
        raise
    finally:
        save(dict(exit_code=code,error=error,elapsed_seconds=time.monotonic()-start),folder/'completion.json')
        save(dict(sha256=payload(folder)), folder/'output_hashes.json')


if __name__ == '__main__':
    main()
