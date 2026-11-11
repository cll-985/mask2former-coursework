"""Recompute the original frozen test comparison with its historical policy."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path.cwd()))
sys.path.append(str(Path.cwd() / 'offline_tools'))
from mva_operating_points import sha256
from mva_full_coco_bootstrap import Resampler, full_ap
from run_mva_e09_bootstrap import MatchingCache


TEST_ROOT = Path('output_offline/Q3_empirical_evidence/M4_frozen_test_20261009_113405')
PINS = {
    'CarDD_COCO/annotations/instances_test2017.json': '130cde81a32820cd83091e78bdbce236753c7675306da004b88bf54bad130a71',
    str(TEST_ROOT/'protocol.json'): '689f745675aee15e4cf85194734576ba6b4cc7f50ef5ef064bc43322b6a0357a',
    str(TEST_ROOT/'fusion/fusion_rule_comparison.json'): '8ea55278b0d3c7788e781bd027ea6eb1746d7eb62dc8deda0f6e69e858bfc30c',
    str(TEST_ROOT/'seed3407/inference/coco_instances_results.json'): '66c4fbb9eab9b10e0efe85cf1b7b7e20e4436880b72543099963ceaead0547c4',
    str(TEST_ROOT/'seed42/inference/coco_instances_results.json'): '4c1b59d566cdced749d9588430d0b994b923818f454760f343cce4c5c2a02375',
    str(TEST_ROOT/'fusion/mask_nms_coco_instances_results.json'): '0cc9752d85cd95b234a9a1037c34f1aed3774982b4ba7f07dd71220e44c5ba7f',
}


def run(output):
    for path, pin in PINS.items():
        assert sha256(path) == pin, path
    output.mkdir()
    dataset = json.loads(Path('CarDD_COCO/annotations/instances_test2017.json').read_text())
    assert len(dataset['images']) == 374 and len(dataset['annotations']) == 785
    assert min(annotation['id'] for annotation in dataset['annotations']) > 0
    names = ('seed3407', 'seed42', 'hard')
    paths = [TEST_ROOT/'seed3407/inference/coco_instances_results.json',
             TEST_ROOT/'seed42/inference/coco_instances_results.json',
             TEST_ROOT/'fusion/mask_nms_coco_instances_results.json']
    rng = np.random.RandomState(20261109)
    small_draw = np.random.RandomState(63).randint(0, 374, size=96)
    caches, points, witness = [], {}, {}
    for name, path in zip(names, paths):
        values = json.loads(path.read_text())
        observed = full_ap(dataset, values)
        cache = MatchingCache(dataset, values)
        assert abs(cache.ap(list(range(374))) - observed) < 1e-8
        resampler = Resampler(dataset, [values])
        assert tuple(resampler.ids) == cache.image_ids
        sample, systems = resampler.draw(small_draw)
        native, cached = full_ap(sample, systems[0]), cache.ap(small_draw)
        assert abs(native - cached) < 1e-8
        points[name] = observed
        witness[name] = dict(native_draw_ap=native, cached_draw_ap=cached)
        caches.append(cache)
        print(name, observed, flush=True)
        del values, resampler, sample, systems
    assert abs(points['hard'] - 52.76531081225052) < 1e-8
    assert abs(points['seed3407'] - 51.0354) < .0001
    assert abs(points['seed42'] - 50.8553) < .0001
    rows = []
    with (output/'draws.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['draw','seed3407_ap','seed42_ap','hard_ap','delta'])
        writer.writeheader()
        for draw in range(1, 2001):
            indices = rng.randint(0, 374, size=374)
            a, b, c = [cache.ap(indices) for cache in caches]
            row = dict(draw=draw, seed3407_ap=a, seed42_ap=b, hard_ap=c, delta=c-a)
            rows.append(row)
            writer.writerow(row)
            if draw % 200 == 0:
                handle.flush()
                print('DRAW', draw, flush=True)
    delta = [row['delta'] for row in rows]
    for path, pin in PINS.items():
        assert sha256(path) == pin, path
    report = dict(status='complete', historical_test_recovery=True, retrospective_interval=True,
                  points=points, original_reference='seed3407', score_floor=None, draws=2000,
                  delta_AP=points['hard']-points['seed3407'],
                  ci95=np.percentile(delta,[2.5,97.5]).tolist(), input_sha256=PINS,
                  equivalence_witness=witness, source_sha256=sha256(__file__),
                  reference_sha256=sha256(Path(__file__).with_name('run_mva_e09_bootstrap.py')),
                  final_input_pins_passed=True)
    with (output/'report.json').open('x') as handle:
        json.dump(report, handle, indent=2)
    print('ORIGINAL_TEST_RECOVERY_COMPLETE', flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
