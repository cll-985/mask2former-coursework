#!/usr/bin/env python3
"""GPU0 shared sequential queue; requires a passing, source-matched CUDA profile."""
import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'offline_tools'))
from mva_shared_gpu_policy import SharingPolicy

UUID = 'GPU-a332beaa-9bf1-c84b-f27c-8bdcaaebed4c'
IMAGE = 'sha256:9253a5d9fe1fe87924d97262d50c68fd7bd13bc2467da1019658286c2524d7b2'


def command_output(command):
    return subprocess.check_output(command, universal_newlines=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path,
        default=Path('output_offline/R01_MVA_MaskRCNN_R50_20261016_shared_gpu0_v1'))
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    profile = args.profile_dir.resolve()
    profile.relative_to(ROOT / 'output_offline')
    output = args.output_root.resolve()
    output.relative_to(ROOT / 'output_offline')
    report = json.loads((profile / 'shared_profile_report.json').read_text())
    manifest = json.loads((profile / 'launch_manifest.json').read_text())
    policy = SharingPolicy()
    assert report['status'] == 'passed' and report['amp_training_steps'] == 20
    assert report['train_image_size'] == 1024 and report['batch_size'] == 2
    assert report['policy'] == asdict(policy) and manifest['gpu_uuid'] == UUID
    assert (profile / 'exit_code.txt').read_text().strip() == '0'
    for path, expected in manifest['source_sha256'].items():
        assert sha256(ROOT / path) == expected, 'Source changed after profile: ' + path
    assert sha256(ROOT / '.runtime/mva_tv0100cu111/torchvision/_C.so') == manifest['runtime']['binary_sha256']
    assert command_output(['docker', 'inspect', '--format', '{{.Image}}', 'm2f_gpu0']) == IMAGE
    exposed = command_output(['docker', 'exec', 'm2f_gpu0', 'nvidia-smi',
                              '--query-gpu=uuid', '--format=csv,noheader'])
    assert UUID in exposed.splitlines(), 'Wrong container GPU mapping'

    def free_mb():
        return int(command_output(['nvidia-smi', '--id=' + UUID,
            '--query-gpu=memory.free', '--format=csv,noheader,nounits']))

    available = free_mb()
    compute_rows = command_output(['nvidia-smi', '--id=' + UUID,
        '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    # Read-only diagnosis found that this queue's board-wide budget guard exits
    # when our ~4.9GiB process overlaps it. Kernel-profile success is not consent
    # to repeatedly interrupt that queue. Keep it intact; do not bypass its guard.
    incompatible_processes = [row for row in compute_rows.splitlines()
                              if '/RSPrompter/' in row]
    if shutil.disk_usage(str(ROOT)).free < 30 * 1024**3:
        raise RuntimeError('Need30GiB free project disk')
    if output.exists():
        raise RuntimeError('Output root exists, refusing duplicate run')
    status = ('blocked_existing_budget_guard' if incompatible_processes else
              'shared_ready' if policy.start_allowed(available) else 'blocked_memory_reserve')
    state = {'status': status,
             'policy': asdict(policy), 'gpu_uuid': UUID, 'container': 'm2f_gpu0',
             'free_gpu_mb': available, 'profile_dir': str(profile),
             'source_sha256': manifest['source_sha256'], 'created_at': now(),
             'pid': os.getpid(), 'test_split_access': False,
             'incompatible_processes': incompatible_processes,
             'jobs': [], 'pending_seeds': [42, 3407]}
    print(json.dumps(state, indent=2), flush=True)
    if not args.launch:
        return
    if incompatible_processes:
        raise RuntimeError('Active RSPrompter board-budget guard is incompatible with this shared run; use sequential scheduling')
    if not policy.start_allowed(available):
        raise RuntimeError('Insufficient reserve, not bypassed by sharing authorization')
    if args.detach:
        log_path = output.with_name(output.name + '_launcher.log')
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--profile-dir', str(profile), '--output-root', str(output), '--launch']
        with log_path.open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'detached_launcher_pid': child.pid, 'log': str(log_path)}), flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)

    def update():
        temporary = output / 'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(str(temporary), str(output / 'status.json'))

    try:
        update()
        for seed in (42, 3407):
            current_processes = command_output(['nvidia-smi', '--id=' + UUID,
                '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
            if '/RSPrompter/' in current_processes:
                raise RuntimeError('RSPrompter became active before seed' + str(seed))
            if not policy.start_allowed(free_mb()):
                raise RuntimeError('Insufficient memory before seed' + str(seed))
            folder = Path('/workspace/Mask2Former-main') / output.relative_to(ROOT) / ('seed' + str(seed))
            command = ['docker', 'exec', '-e', 'CUDA_VISIBLE_DEVICES=' + UUID,
                       '-e', 'OMP_NUM_THREADS=2', '-e', 'MKL_NUM_THREADS=2',
                       '-e', 'OPENBLAS_NUM_THREADS=1', '-w', '/workspace/Mask2Former-main',
                       'm2f_gpu0', 'nice', '-n', '10', 'python', '-u',
                       'offline_tools/train_mva_maskrcnn_shared.py', '--seed', str(seed),
                       '--output-dir', str(folder)]
            entry = {'seed': seed, 'status': 'running', 'command': command, 'started_at': now()}
            state['jobs'].append(entry)
            state['status'] = 'running'
            with (output / ('seed{}_console.log'.format(seed))).open('x') as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                entry['host_docker_exec_pid'] = process.pid
                update()
                code = process.wait()
            entry.update({'exit_code': code, 'finished_at': now(),
                          'status': 'complete' if code == 0 else 'stopped'})
            update()
            if code:
                raise RuntimeError('seed{} stopped, inspect checkpoint/log before resume'.format(seed))
            state['pending_seeds'].remove(seed)
            update()
        state.update({'status': 'training_complete_predictions_ready', 'finished_at': now()})
        (output / 'exit_code.txt').write_text('0\n')
    except BaseException as exc:
        state.update({'status': 'stopped_requires_inspection', 'error': repr(exc),
                      'finished_at': now()})
        (output / 'exit_code.txt').write_text('1\n')
        raise
    finally:
        update()


if __name__ == '__main__':
    main()
