"""Paired full-COCO segmentation AP bootstrap for the frozen E09 comparison."""

import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mva_full_coco_bootstrap import Resampler, full_ap, interval, mask_predictions

BUNDLE = Path('/workspace/e09')
GT = BUNDLE / 'cohort/instances_e09_positive_ann_ids.json'
RESULTS = BUNDLE / 'results'
OUTPUT = BUNDLE / 'statistics/paired_full_coco_20261108'
SEED = 20261105
DRAWS = 2000
FIELDS = ('draw', 'seed42_ap', 'seed2026_ap', 'hard_nms_ap', 'reference_ap',
          'delta', 'complete_categories')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def memory_guard():
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if available < 8 * 1024**2 or peak > 14 * 1024**2:
        raise MemoryError('E09 statistics resource limit reached')


class MatchingCache:
    """Reuse per-image matches and rerun native accumulation on every draw."""

    def __init__(self, dataset, predictions):
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = copy.deepcopy(dataset)
            gt.createIndex()
            if predictions:
                dt = gt.loadRes(mask_predictions(predictions))
            else:
                dt = COCO()
                dt.dataset = {'images': gt.dataset['images'], 'categories': gt.dataset['categories'],
                              'annotations': []}
                dt.createIndex()
            evaluator = COCOeval(gt, dt, 'segm')
            evaluator.params.areaRng = [evaluator.params.areaRng[0]]
            evaluator.params.areaRngLbl = ['all']
            evaluator.params.maxDets = [100]
            evaluator.evaluate()
        self.image_ids = tuple(evaluator.params.imgIds)
        self.matches = np.asarray(evaluator.evalImgs, dtype=object).reshape(
            len(evaluator.params.catIds), len(self.image_ids))
        self.params = copy.deepcopy(evaluator.params)

    def ap(self, indices):
        # Native COCO concatenates image records in image-ID order, then sorts
        # scores stably. Fresh draw-order IDs preserve that ordering for ties.
        evaluator = COCOeval(iouType='segm')
        evaluator.params = copy.deepcopy(self.params)
        evaluator.params.imgIds = list(range(1, len(indices) + 1))
        evaluator._paramsEval = copy.deepcopy(evaluator.params)
        evaluator.evalImgs = self.matches[:, indices].reshape(-1).tolist()
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.accumulate()
        precision = evaluator.eval['precision']
        valid = precision[precision > -1]
        if not valid.size:
            raise ValueError('No evaluable category in draw')
        return float(valid.mean() * 100)


def fixture():
    mask = np.zeros((8, 8), dtype=np.uint8, order='F')
    mask[1:5, 1:5] = 1
    rle = mask_util.encode(mask)
    miss = np.zeros_like(mask, order='F')
    miss[5:, 5:] = 1
    miss_rle = mask_util.encode(miss)
    images = [dict(id=i, width=8, height=8) for i in (3, 8, 9)]
    categories = [dict(id=i, name=str(i)) for i in (1, 2)]
    annotations = [
        dict(id=1, image_id=3, category_id=1, segmentation=rle, area=16, iscrowd=0),
        dict(id=2, image_id=8, category_id=1, segmentation=rle, area=16, iscrowd=1),
        dict(id=3, image_id=9, category_id=2, segmentation=rle, area=16, iscrowd=0),
    ]
    dataset = dict(images=images, categories=categories, annotations=annotations)
    predictions = [dict(image_id=i, category_id=c, score=.5, segmentation=r)
                   for i, c, r in [(3, 1, rle), (8, 1, rle), (8, 1, miss_rle),
                                   (9, 1, miss_rle), (9, 2, rle)]]
    resampler = Resampler(dataset, [predictions])
    cache = MatchingCache(dataset, predictions)
    for draw in ([0, 1, 2], [2, 1, 0], [2, 0, 2], [0, 1, 1, 0]):
        sample, systems = resampler.draw(draw)
        assert abs(cache.ap(draw) - full_ap(sample, systems[0])) < 1e-10


def cached_inputs(dataset, paths, observed, rng):
    caches = []
    witness = []
    # Compare one full-size random draw and one small repeated-image draw
    # to the independent path that physically clones images and annotations.
    full = rng.randint(0, len(dataset['images']), size=len(dataset['images']))
    small = rng.randint(0, len(dataset['images']), size=96)
    for path, expected in zip(paths, observed):
        values = [x for x in read(path) if x['score'] >= .001]
        resampler = Resampler(dataset, [values])
        cache = MatchingCache(dataset, values)
        assert cache.image_ids == tuple(resampler.ids)
        original = cache.ap(list(range(len(cache.image_ids))))
        assert abs(original - expected) < 1e-8, (path, original, expected)
        comparisons = []
        for indices in (small, full):
            sample, systems = resampler.draw(indices)
            native = full_ap(sample, systems[0])
            cached = cache.ap(indices)
            assert abs(cached - native) < 1e-8, (path, cached, native)
            comparisons.append({'images': len(indices), 'native_ap': native, 'cached_ap': cached})
            del sample, systems
        witness.append({'file': str(path), 'replayed_point_ap': original, 'draws': comparisons})
        caches.append(cache)
        memory_guard()
        print('NATIVE_EQUIVALENCE_PASS', path, flush=True)
    save(OUTPUT / 'equivalence_witness.json', witness)
    return caches


def recover(path):
    if not path.exists():
        return []
    content = path.read_bytes()
    end = content.rfind(b'\n') + 1
    if end != len(content):
        with path.open('r+b') as handle:
            handle.truncate(end)
            handle.flush()
            os.fsync(handle.fileno())
    reader = csv.DictReader(io.StringIO(content[:end].decode()))
    rows = list(reader)
    if end:
        assert tuple(reader.fieldnames) == FIELDS
    for index, row in enumerate(rows):
        assert int(row['draw']) == index
        a, b, c, ref, delta = (float(row[k]) for k in FIELDS[1:6])
        assert np.isfinite([a, b, c, ref, delta]).all()
        assert ref == max(a, b) and abs(delta - (c - ref)) < 1e-12
        assert row['complete_categories'] in ('0', '1')
    assert len(rows) <= DRAWS
    return rows


def run(resume):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    fixture()
    dataset = read(GT)
    assert len(dataset['images']) == 6704 and len(dataset['categories']) == 9
    assert min(x['id'] for x in dataset['annotations']) > 0
    assert not any(k in image for image in dataset['images']
                   for k in ('vehicle_id', 'sequence_id', 'video_id', 'group_id'))
    assert sha(GT) == 'aa3db07e39f7b88c4773ae4cb6b5f375fb5406bc39ddb389b6d15658300f6c49'
    singles = [RESULTS / ('seed' + seed) for seed in ('42', '2026')]
    paths = [p / 'inference/coco_instances_results.json' for p in singles]
    paths.append(RESULTS / 'controls/mask_nms.json')
    report_path = RESULTS / 'controls_report.json'
    controls = read(report_path)
    observed = [controls['singles'][seed]['AP'] for seed in ('42', '2026')]
    observed.append(controls['controls']['mask_nms']['metrics']['AP'])
    input_hashes = {str(p): sha(p) for p in [GT, report_path] + paths}
    for p, folder in zip(paths, singles):
        result = read(folder / 'result.json')
        assert result['images'] == 6704 and result['prediction_sha256'] == input_hashes[str(p)]
    contract = {'inputs': input_hashes, 'script_sha256': sha(Path(__file__)),
                'reference_sha256': sha(Path(__file__).with_name('mva_full_coco_bootstrap.py')),
                'numpy_version': np.__version__,
                'cocoeval_sha256': sha(Path(sys.modules[COCOeval.__module__].__file__)),
                'seed': SEED, 'draws': DRAWS, 'floor': .001, 'observed_aps': observed,
                'primary_rule': 'mask_nms', 'resampling_unit': 'image',
                'vehicle_groups_available': False}
    OUTPUT.mkdir(parents=True, exist_ok=resume)
    with (OUTPUT / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = OUTPUT / 'contract.json'
        if resume:
            assert read(protocol) == contract
        else:
            save(protocol, contract)
        save(OUTPUT / 'progress.json', {'status': 'validating', 'target': DRAWS, 'pid': os.getpid()})
        try:
            bootstrap(dataset, paths, observed)
        except BaseException as error:
            save(OUTPUT / 'failure.json', {'error': repr(error), 'pid': os.getpid()})
            raise


def bootstrap(dataset, paths, observed):
    caches = cached_inputs(dataset, paths, observed, np.random.RandomState(20261108))
    resampler = Resampler(dataset, [])
    path = OUTPUT / 'draws.csv'
    rows = recover(path)
    rng = np.random.RandomState(SEED)
    start = time.monotonic()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        for draw in range(DRAWS):
            indices = rng.randint(0, len(resampler.ids), size=len(resampler.ids))
            if draw < len(rows):
                continue
            memory_guard()
            a, b, c = (cache.ap(indices) for cache in caches)
            ref = max(a, b)
            row = dict(draw=draw, seed42_ap=a, seed2026_ap=b, hard_nms_ap=c,
                       reference_ap=ref, delta=c-ref,
                       complete_categories=int(resampler.complete_categories(indices)))
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            rows.append(row)
            progress = {'status': 'running', 'completed': len(rows), 'target': DRAWS,
                        'elapsed_this_session_seconds': time.monotonic() - start, 'pid': os.getpid()}
            save(OUTPUT / 'progress.json', progress)
            if len(rows) == 1 or len(rows) % 25 == 0:
                print('E09_BOOTSTRAP_PROGRESS', json.dumps(progress), flush=True)
    assert len(rows) == DRAWS
    for name, expected in read(OUTPUT / 'contract.json')['inputs'].items():
        assert sha(Path(name)) == expected, name
    complete = [float(r['delta']) for r in rows if int(r['complete_categories'])]
    save(OUTPUT / 'bootstrap_report.json',
         {'status': 'complete', 'draws': DRAWS, 'seed': SEED,
          'point_delta': observed[2] - max(observed[:2]),
          'complete_category_conditional_interval': interval(complete),
          'unfiltered_interval': interval([float(r['delta']) for r in rows]),
          'missing_category_draws': DRAWS-len(complete),
          'reference_reselected_every_draw': True, 'vehicle_groups_available': False,
          'draws_sha256': sha(path)})
    save(OUTPUT / 'progress.json', {'status': 'complete', 'completed': DRAWS, 'target': DRAWS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        fixture()
        print('COCO_CACHE_SELF_TEST_PASS')
    else:
        run(args.resume)
