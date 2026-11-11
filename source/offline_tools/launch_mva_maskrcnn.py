#!/usr/bin/env python3
"""Host-side stdlib launcher. Default is read-only preflight, never wait/kill."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
CONTAINER_ROOT = Path('/workspace/Mask2Former-main')
EXPECTED_IMAGE = 'sha256:9253a5d9fe1fe87924d97262d50c68fd7bd13bc2467da1019658286c2524d7b2'
GPU1_EXISTING_IMAGE = 'sha256:c35f519fe286f089f9728277fadcd8991ead8c80038cf0026f851292bdd7b97a'
TRACKED = ['configs/cardd/mva_mask_rcnn_R50_fpn_replication.yaml',
           'offline_tools/train_mva_maskrcnn.py',
           'refine-logs/MVA_ARCH_REPLICATION_PROTOCOL_20261016.md',
           '.aris/compute/remote-maskrcnn.md']


def run(command):
    return subprocess.check_output(command, universal_newlines=True).strip()


def gpu_inventory(prefix=()):
    text = run(list(prefix) + ['nvidia-smi',
               '--query-gpu=index,uuid,memory.used,utilization.gpu',
               '--format=csv,noheader,nounits'])
    return [dict(zip(('index', 'uuid', 'memory_used_mb', 'utilization_percent'),
                     (x.strip() for x in row.split(','))))
            for row in text.splitlines() if row.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', choices=('m2f_gpu3', 'm2f_gpu0',
                         'm2f_container'), default='m2f_gpu3')
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--smoke-dir', required=True, type=Path)
    parser.add_argument('--output-root', type=Path,
                        default=Path('output_offline/R01_MVA_MaskRCNN_R50_20261016_v1'))
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    output = args.output_root.resolve()
    output.relative_to(ROOT / 'output_offline')
    smoke_dir = args.smoke_dir.resolve()
    smoke_dir.relative_to(ROOT / 'output_offline')
    report = json.loads((smoke_dir / 'smoke_report.json').read_text())
    manifest = json.loads((smoke_dir / 'launch_manifest.json').read_text())
    assert report['status'] == 'passed' and report['checkpoint_reload_predictions_exact']
    for path in TRACKED:
        assert digest(ROOT / path) == manifest['source_sha256'][path], 'Changed since smoke: ' + path
    assert digest(ROOT / '.runtime/mva_tv0100cu111/torchvision/_C.so') == manifest['runtime']['binary_sha256']
    image = run(['docker', 'inspect', '--format', '{{.Image}}', args.container])
    expected_image = GPU1_EXISTING_IMAGE if args.container == 'm2f_container' else EXPECTED_IMAGE
    assert image == expected_image, 'Container image differs from validated image'
    host_gpus = gpu_inventory()
    container_gpus = gpu_inventory(['docker', 'exec', args.container])
    selected = [x for x in host_gpus if x['uuid'] == args.gpu_uuid]
    assert len(selected) == 1, 'Unknown host GPU UUID'
    assert any(x['uuid'] == args.gpu_uuid for x in container_gpus), 'GPU not exposed to container'
    idle = int(selected[0]['memory_used_mb']) < 500 and int(selected[0]['utilization_percent']) <= 10
    free_gb = shutil.disk_usage(str(ROOT)).free / 1024**3
    state = {'status': 'preflight_ready' if idle else 'blocked_gpu_busy',
             'host_gpus': host_gpus, 'container_gpus': container_gpus,
             'gpu_uuid': args.gpu_uuid, 'container': args.container,
             'container_image': image, 'free_disk_gb': free_gb,
             'seeds': [42, 3407], 'sources_sha256': {p: digest(ROOT / p) for p in TRACKED},
             'cpu_smoke': str(smoke_dir), 'jobs': [], 'test_split_access': False}
    state['kernel_witness'] = str(smoke_dir)
    state['kernel_witness_device'] = report.get('device', 'unspecified')
    deployment_ledger = ROOT / '.aris/compute/mva-maskrcnn-gpu1.md'
    if args.container == 'm2f_container':
        state['deployment_ledger_sha256'] = digest(deployment_ledger)
    print(json.dumps(state, indent=2), flush=True)
    if not args.launch:
        return
    if not idle:
        raise RuntimeError('GPU is busy: refusing training, waiting loop or process termination')
    if free_gb < 30:
        raise RuntimeError('Need at least 30 GiB free project disk')
    if output.exists():
        raise RuntimeError('Output root exists; refusing duplicate launch')
    if args.detach:
        log_path = output.with_name(output.name + '_launcher.log')
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--container', args.container, '--gpu-uuid', args.gpu_uuid,
                   '--smoke-dir', str(smoke_dir), '--output-root', str(output), '--launch']
        with log_path.open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'detached_launcher_pid': child.pid, 'log': str(log_path)}), flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)

    def update():
        tmp = output / 'status.json.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(str(tmp), str(output / 'status.json'))

    def execute(name, seed, smoke=False):
        # Recheck immediately before each process, not just once at launcher startup.
        gpu = next(x for x in gpu_inventory() if x['uuid'] == args.gpu_uuid)
        # NVML utilization can retain a just-completed job's final sampling window.
        # Only allow a short transition with <500MiB used; never wait on a busy job.
        if state['jobs'] and state['jobs'][-1].get('status') == 'complete':
            for _ in range(3):
                if int(gpu['memory_used_mb']) >= 500 or int(gpu['utilization_percent']) <= 10:
                    break
                time.sleep(1)
                gpu = next(x for x in gpu_inventory() if x['uuid'] == args.gpu_uuid)
        if int(gpu['memory_used_mb']) >= 500 or int(gpu['utilization_percent']) > 10:
            raise RuntimeError('GPU became busy before ' + name)
        folder = CONTAINER_ROOT / output.relative_to(ROOT) / name
        command = ['docker', 'exec', '-e', 'CUDA_VISIBLE_DEVICES=' + args.gpu_uuid,
                   '-e', 'OMP_NUM_THREADS=2', '-e', 'MKL_NUM_THREADS=2',
                   '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OPENCV_FOR_THREADS_NUM=2',
                   '-w', str(CONTAINER_ROOT), args.container,
                   'python', '-u', 'offline_tools/train_mva_maskrcnn.py',
                   '--seed', str(seed), '--output-dir', str(folder)]
        if smoke:
            command += ['--smoke-only', '--smoke-device', 'cuda']
        entry = {'name': name, 'seed': seed, 'status': 'running', 'command': command,
                 'started_at': datetime.datetime.now().astimezone().isoformat()}
        state['jobs'].append(entry)
        state['status'] = 'running'
        with (output / (name + '_console.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
            entry['host_docker_exec_pid'] = process.pid
            update()
            code = process.wait()
        entry.update({'exit_code': code, 'status': 'complete' if code == 0 else 'failed',
                      'finished_at': datetime.datetime.now().astimezone().isoformat()})
        update()
        if code:
            raise RuntimeError('{} exited {}; inspect log, no auto-retry'.format(name, code))

    try:
        update()
        execute('gpu_smoke_seed42', 42, smoke=True)
        for seed in (42, 3407):
            execute('seed' + str(seed), seed)
        state['status'] = 'training_complete_predictions_ready'
        (output / 'exit_code.txt').write_text('0\n')
    except Exception as exc:
        state.update({'status': 'stopped_requires_inspection', 'error': repr(exc)})
        (output / 'exit_code.txt').write_text('1\n')
        raise
    finally:
        update()


if __name__ == '__main__':
    main()
