#!/usr/bin/env python3
"""Launch frozen CPU-only MVA phase2 into a new output directory."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.compare_fusion_rules import atomic_json_dump

BASES = {
    '42': 'output_offline/B00_formal_R50/20260819_133145_seed42/inference/coco_instances_results.json',
    '3407': 'output_offline/B00_formal_R50/20260822_073123_seed3407/inference/coco_instances_results.json',
    '2026': 'output_offline/B00_formal_R50/20260824_112238_seed2026/inference/coco_instances_results.json'}
MAIN_NMS = 'output_offline/P10_fusion_rule_controls_R50/baseline42_baseline3407/mask_nms_coco_instances_results.json'
GT = 'CarDD_COCO/annotations/instances_val2017.json'
TZ = datetime.timezone(datetime.timedelta(hours=8))


def now():
    return datetime.datetime.now(TZ).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    os.chdir(str(ROOT))
    output = (ROOT / args.output_root).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    for name in list(BASES.values()) + [GT, MAIN_NMS]:
        if not Path(name).is_file():
            raise RuntimeError('Missing input: ' + name)
    protocol_path = ROOT / 'refine-logs/MVA_PHASE2_PROTOCOL_20261015.md'
    state = {'started_at': now(), 'pid': os.getpid(), 'status': 'running',
             'output_root': str(output), 'tasks': {}, 'cpu_workers': 2,
             'gpu_visible_devices': '', 'test_split_access': False,
             'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    lock = threading.Lock()

    def update(name=None, values=None):
        with lock:
            if name:
                state['tasks'][name] = values
                if values.get('status') == 'failed':
                    state['status'] = 'running_with_failed_tasks'
            atomic_json_dump(state, str(output / 'status.json'), indent=2)

    def run(name, script, params):
        command = [sys.executable, '-u', 'offline_tools/' + script] + params
        entry = {'started_at': now(), 'status': 'running', 'command': command}
        with (output / (name + '_console.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            entry['pid'] = process.pid
            update(name, dict(entry))
            code = process.wait()
        entry.update({'exit_code': code, 'finished_at': now(),
                      'status': 'complete' if code == 0 else 'failed'})
        update(name, dict(entry))
        if code:
            raise RuntimeError('Task {} exited {}'.format(name, code))

    def controls_chain():
        for a, b, reference, consensus_ap in (
                ('42', '3407', '3407', 55.37573754120204),
                ('42', '2026', '42', 54.95448082195746),
                ('3407', '2026', '3407', 55.23732500101977)):
            name = 'seed{}_seed{}'.format(a, b)
            folder = str(output / 'controls' / name)
            params = ['--gt-json', GT, '--base-results', BASES[a], '--other-results', BASES[b],
                      '--output-dir', folder, '--expected-consensus-ap', str(consensus_ap)]
            if (a, b) == ('42', '3407'):
                params.extend(['--expected-mask-nms-ap', '55.570616180942466'])
            run(name + '_controls', 'mva_fusion_controls.py', params)
            run(name + '_mechanisms', 'mva_fusion_mechanisms.py', [
                '--gt-json', GT, '--left-results', BASES[a], '--right-results', BASES[b],
                '--reference-results', BASES[reference], '--controls-dir', folder,
                '--output-dir', str(output / 'mechanisms' / name)])

    def bootstrap_chain():
        run('primary_full_coco_bootstrap', 'mva_full_coco_bootstrap.py', [
            '--gt-json', GT, '--baseline-results', BASES['3407'],
            '--candidate-results', MAIN_NMS, '--output-dir', str(output / 'primary_bootstrap'),
            '--resamples', '2000', '--seed', '20261015',
            '--expected-baseline-ap', '54.48288309947704',
            '--expected-candidate-ap', '55.570616180942466'])

    update()
    try:
        run('correctness_tests', 'test_mva_phase2.py', [])
        # Two independent single-thread CPU workers; no GPU utilization.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(controls_chain), pool.submit(bootstrap_chain)]
            errors = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(repr(exc))
            if errors:
                raise RuntimeError('; '.join(errors))
        state.update({'status': 'complete', 'finished_at': now()})
        (output / 'exit_code.txt').write_text('0\n')
    except Exception as exc:
        state.update({'status': 'failed', 'error': repr(exc), 'finished_at': now()})
        (output / 'exit_code.txt').write_text('1\n')
        raise
    finally:
        update()


if __name__ == '__main__':
    main()
