#!/usr/bin/env python3
"""Explicit GPU2 cooperative seed3407 deployment. Default: read-only preflight."""
import argparse
import csv
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / '.runtime_gpu2_native_v2/conda'
UUID = 'GPU-226c3a72-1bdd-588a-65d8-7a6ea4d8571a'
EXTERNAL_PID = 12421
ALLOCATOR_MB, CONTEXT_MB, RESERVE_MB = 6144, 2048, 4096
DUTY = 0.40
TRACKED = ['offline_tools/run_mva_gpu2_native.py',
           '.aris/compute/mva-maskrcnn-gpu2-fast.md',
           'configs/cardd/mva_mask_rcnn_R50_fpn_replication.yaml',
           'offline_tools/train_mva_maskrcnn.py',
           'refine-logs/MVA_ARCH_REPLICATION_PROTOCOL_20261016.md',
           '.aris/compute/remote-maskrcnn.md']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(value, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(str(tmp), str(path))


def output(command):
    return subprocess.check_output(command, universal_newlines=True).strip()


def inventory():
    row = output(['nvidia-smi', '-i', UUID, '--query-gpu=uuid,memory.free',
                  '--format=csv,noheader,nounits']).split(',')
    if row[0].strip() != UUID:
        raise RuntimeError('Wrong GPU identity')
    processes = output(['nvidia-smi', '-i', UUID,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    return int(row[1].strip()), {int(x.strip()) for x in processes.splitlines() if x.strip()}


def verify_sources():
    witness = ROOT / 'output_offline/R01_MVA_GPU1_preflight_20261017_v1/launch_manifest.json'
    m = json.loads(witness.read_text())
    for p, expected in m['source_sha256'].items():
        if sha(ROOT / p) != expected:
            raise RuntimeError('Frozen source changed: ' + p)
    binary = ROOT / '.runtime/mva_tv0100cu111/torchvision/_C.so'
    if sha(binary) != m['runtime']['binary_sha256']:
        raise RuntimeError('Private binary changed')
    return {p: sha(ROOT / p) for p in TRACKED}


def native_env(mode, folder):
    env = dict(os.environ)
    env.update({'CUDA_VISIBLE_DEVICES': '' if mode == 'cpu' else UUID,
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '1', 'OPENCV_FOR_THREADS_NUM': '2',
        'MPLCONFIGDIR': str(folder / 'mpl_cache'),
        'PATH': str(ENV / 'bin') + ':' + env.get('PATH', ''),
        'LD_LIBRARY_PATH': ':'.join([str(ENV / 'lib'),
            str(ENV / 'lib/python3.7/site-packages/torch/lib')])})
    return env


def launch(args):
    folder = args.output_dir.resolve()
    folder.relative_to(ROOT / 'output_offline')
    if folder.exists():
        raise RuntimeError('Output exists; no overwrite or duplicate launch')
    copy = json.loads((ENV.parent / 'copy_status.json').read_text())
    if copy['source_exit'] or copy['extract_exit'] or not (ENV / 'bin/python').is_file():
        raise RuntimeError('Runtime copy incomplete')
    hashes = verify_sources()
    if args.mode != 'cpu':
        free, pids = inventory()
        if pids != {EXTERNAL_PID} or free < ALLOCATOR_MB + CONTEXT_MB + RESERVE_MB:
            raise RuntimeError('External PID changed/extra job/insufficient headroom')
        if args.cpu_witness is None:
            raise RuntimeError('Need new native CPU witness')
        check_evidence(args.cpu_witness, 'cpu', hashes)
    if args.mode == 'train':
        if args.profile_dir is None:
            raise RuntimeError('Need source-matched GPU2 profile')
        check_evidence(args.profile_dir, 'profile', hashes)
    state = {'status': 'ready', 'mode': args.mode, 'seed': 3407,
             'gpu_uuid': UUID, 'external_pid': EXTERNAL_PID,
             'source_sha256': hashes, 'test_split_access': False,
             'allocator_mb': ALLOCATOR_MB, 'reserve_mb': RESERVE_MB,
             'context_allowance_mb': CONTEXT_MB, 'duty_cycle': DUTY,
             'native_python_sha256': sha(ENV / 'bin/python')}
    print(json.dumps(state, indent=2), flush=True)
    if not args.launch:
        return
    folder.mkdir(parents=True, exist_ok=False)
    save(state, folder / 'deployment.json')
    save({'shared_duty': DUTY, 'checkpoint_and_stop': False}, folder / 'control.json')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    command = ['ionice', '-c', '2', '-n', '7', 'nice', '-n', '10', 'taskset',
        '-c', ','.join(str(x) for x in cpus), str(ENV / 'bin/python'), '-u',
        str(Path(__file__).resolve()), '--worker', '--mode', args.mode,
        '--output-dir', str(folder)]
    with (folder / 'console.log').open('x') as log:
        proc = subprocess.Popen(command, cwd=str(ROOT), env=native_env(args.mode, folder),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    state.update(status='launched', pid=proc.pid, cpu_affinity=cpus, command=command)
    save(state, folder / 'deployment.json')
    print(json.dumps({'pid': proc.pid, 'folder': str(folder)}), flush=True)


def check_evidence(folder, mode, hashes):
    folder = folder.resolve()
    folder.relative_to(ROOT / 'output_offline')
    state = json.loads((folder / 'deployment.json').read_text())
    report = json.loads((folder / 'completion.json').read_text())
    if (state['mode'] != mode or state['source_sha256'] != hashes or report['exit_code'] != 0
            or state['native_python_sha256'] != sha(ENV / 'bin/python')):
        raise RuntimeError('Evidence failed or sources changed')
    key = 'smoke_report.json' if mode == 'cpu' else 'shared_profile_report.json'
    result = json.loads((folder / 'stage' / key).read_text())
    if result['status'] != 'passed':
        raise RuntimeError('Kernel witness failed')


class Pressure(RuntimeError):
    pass


def external_progress():
    # User-authorized exact resource log only. Never consume its terminal or data.
    code = ("import csv,json; p='/workspace/yolo11/ultralytics/runs/detect/train-30/results.csv'; "
            "r=list(csv.DictReader(open(p))); print(json.dumps([{'epoch':int(x['epoch']),"
            "'time':float(x['time'])} for x in r[-4:]]))")
    return json.loads(output(['docker', 'exec', 'detrts', 'python', '-c', code]))


def external_slow(rows, starting_epoch):
    # Two wholly overlapping completed epoch intervals; conservative safety rule,
    # not a causal claim about slowdown. Baseline recent epochs~1037..1044sec.
    intervals = [b['time']-a['time'] for a,b in zip(rows,rows[1:])
                 if a['epoch'] > starting_epoch]
    return len(intervals) >= 2 and all(t > 1044 * 1.25 for t in intervals[-2:])


def policy_duty(requested, rows, starting_epoch, exclusive=False):
    if not isinstance(requested, (float, int)) or not math.isfinite(requested) or not 0.20 <= requested <= 0.50:
        raise Pressure('Requested shared duty must be finite0.20..0.50')
    if exclusive:
        return 1.0
    intervals = [b['time']-a['time'] for a,b in zip(rows, rows[1:])
                 if a['epoch'] > starting_epoch]
    if intervals and intervals[-1] > 1044 * 1.10:
        return 0.20
    if len(intervals) >= 2 and all(t <= 1044 * 1.05 for t in intervals[-2:]):
        return 0.50
    return float(requested)


def clean_exclusive(pids, progress, own_pid):
    return pids == {own_pid} and bool(progress) and progress[-1]['epoch'] >= 300


def worker(args):
    if Path(sys.prefix).resolve() != ENV.resolve():
        raise RuntimeError('Wrong native Python prefix')
    hashes = verify_sources()
    folder = args.output_dir.resolve()
    deployment = json.loads((folder / 'deployment.json').read_text())
    if hashes != deployment['source_sha256']:
        raise RuntimeError('Sources changed after launch')
    if deployment['mode'] != args.mode:
        raise RuntimeError('Mode differs from launch record')
    args.output_dir = folder / 'stage'
    args.seed, args.smoke_only, args.smoke_device = 3407, args.mode == 'cpu', 'cpu'
    sys.path.insert(0, str(ROOT / 'offline_tools'))
    import train_mva_maskrcnn as base
    torch = base.torch
    torch.set_num_threads(2)
    args.output_dir.mkdir(exist_ok=False)
    cfg = base.make_cfg(args)
    base.default_setup(cfg, args)
    for p, expected in base.INPUTS.items():
        if base.sha256(ROOT / p) != expected:
            raise RuntimeError('Frozen input changed: ' + p)
    if base.torchvision.__version__ != '0.10.0+cu111':
        raise RuntimeError('Unexpected torchvision')
    Path(base.torchvision.__file__).resolve().relative_to(base.OVERLAY)
    base.register_data()
    save({'source_sha256': hashes, 'input_sha256': base.INPUTS, 'seed': 3407,
          'native_prefix': sys.prefix, 'torch': torch.__version__,
          'torchvision': base.torchvision.__version__, 'test_split_access': False},
         args.output_dir / 'launch_manifest.json')
    if args.mode == 'cpu':
        save(base.smoke(cfg, args.output_dir), args.output_dir / 'smoke_report.json')
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
        raise RuntimeError('Wrong physical GPU binding')
    torch.cuda.set_per_process_memory_fraction(
        ALLOCATOR_MB * 1024**2 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats()
    if args.mode == 'profile':
        cfg = cfg.clone(); cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.DATASETS.TEST = ()
        cfg.freeze()
    last_gate = [0.0]
    last_progress = [0.0]
    initial_progress = external_progress()
    starting_epoch = initial_progress[-1]['epoch']
    progress_rows = [initial_progress]
    effective_duty = [DUTY]
    requested_stop = [None]
    in_updated_step = [False]
    last_completed = [-1]
    signal.signal(signal.SIGUSR1, lambda *_: requested_stop.__setitem__(0, 'Requested checkpoint stop'))
    save({'baseline_rows': initial_progress, 'starting_epoch': starting_epoch,
          'baseline_epoch_seconds': 1044, 'slowdown_stop_fraction': 0.25},
         folder / 'external_progress.json')

    def gate(force=False):
        if not force and time.monotonic() - last_gate[0] < 1.0:
            return
        free, pids = inventory()
        exclusive = False
        if EXTERNAL_PID not in pids:
            # Never infer a clean completion merely from a vanished PID.
            progress_rows[0] = external_progress()
            exclusive = clean_exclusive(pids, progress_rows[0], os.getpid())
            if not exclusive:
                raise Pressure('External exit not verified as clean epoch300 completion')
        if pids - {EXTERNAL_PID, os.getpid()}:
            raise Pressure('Extra GPU task appeared')
        if free < RESERVE_MB:
            raise Pressure('GPU free reserve below4096MiB')
        import psutil
        parent = psutil.Process()
        rss = sum(p.memory_info().rss for p in [parent] + parent.children(recursive=True)
                  if p.is_running())
        if rss > 16 * 1024**3:
            raise Pressure('Our aggregate RSS above16GiB')
        if time.monotonic() - last_progress[0] >= 120:
            try:
                progress = external_progress()
            except Exception as exc:
                raise Pressure('Cannot verify external progress: ' + repr(exc))
            save({'rows': progress, 'starting_epoch': starting_epoch},
                 folder / 'external_latest.json')
            if external_slow(progress, starting_epoch):
                raise Pressure('External epoch time exceeded baseline25% for two full epochs')
            progress_rows[0] = progress
            last_progress[0] = time.monotonic()
        try:
            control = json.loads((folder / 'control.json').read_text())
        except (OSError, ValueError) as exc:
            raise Pressure('Cannot read runtime control: ' + repr(exc))
        effective_duty[0] = policy_duty(control['shared_duty'], progress_rows[0],
                                        starting_epoch, exclusive)
        if control.get('checkpoint_and_stop'):
            requested_stop[0] = 'Control requested checkpoint stop'
        if requested_stop[0] and not in_updated_step[0]:
            raise Pressure(requested_stop[0])
        last_gate[0] = time.monotonic()

    def relinquish(elapsed):
        remaining = elapsed * (1 / effective_duty[0] - 1)
        while remaining > 0:
            tick = min(1.0, remaining); time.sleep(tick); remaining -= tick
            try:
                gate()
            except Pressure as exc:
                requested_stop[0] = str(exc)
                break

    class Trainer(base.ReplicationTrainer):
        def run_step(self):
            in_updated_step[0] = False
            gate()
            torch.cuda.synchronize(); start = time.monotonic()
            super().run_step()
            torch.cuda.synchronize(); elapsed = time.monotonic() - start
            in_updated_step[0] = True
            relinquish(elapsed)
            if (self.iter + 1) % 20 == 0:
                save({'iteration': self.iter, 'active_seconds': elapsed,
                      'duty_cycle': effective_duty[0], 'peak_reserved_mb': torch.cuda.max_memory_reserved()/1024**2,
                      'gpu_free_mb': inventory()[0]}, folder / 'sharing_status.json')

        @classmethod
        def test(cls, test_cfg, model, evaluators=None):
            class CooperativeModel(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.original = model
                def forward(self, inputs):
                    gate(); torch.cuda.synchronize(); tick = time.monotonic()
                    result = self.original(inputs)
                    torch.cuda.synchronize(); relinquish(time.monotonic() - tick)
                    return result
            return super().test(test_cfg, CooperativeModel(), evaluators)

    gate(force=True)
    trainer = Trainer(cfg)
    base.load_verified_pretrain(trainer.model)
    from detectron2.engine.train_loop import HookBase

    class Maintenance(HookBase):
        def after_step(self):
            # Appended AFTER native scheduler/checkpoint/eval/writer hooks, so
            # the saved iteration/optimizer/scaler/LR state share a boundary.
            last_completed[0] = self.trainer.iter
            in_updated_step[0] = False
            if requested_stop[0] or (self.trainer.iter + 1) % 1000 == 0:
                self.trainer.checkpointer.save('runtime_latest', iteration=self.trainer.iter)
            if requested_stop[0]:
                raise Pressure(requested_stop[0])

    if args.mode == 'train':
        trainer.register_hooks([Maintenance()])
    started = time.monotonic()
    try:
        if args.mode == 'train':
            trainer.train()
        else:
            initial = trainer.model.roi_heads.box_predictor.cls_score.weight.detach().clone()
            with base.EventStorage(0) as storage:
                for step in range(20):
                    trainer.iter = step; trainer.run_step(); trainer.scheduler.step(); storage.step()
            in_updated_step[0] = False
            if requested_stop[0]:
                raise Pressure(requested_stop[0])
            if torch.equal(initial, trainer.model.roi_heads.box_predictor.cls_score.weight):
                raise RuntimeError('Profile made no head update')
            records = base.DatasetCatalog.get('mva_maskrcnn_cardd_val')
            records.sort(key=lambda r: (-r['width']*r['height'], r['image_id']))
            mapper = base.DatasetMapper(cfg, False); trainer.model.eval()
            with torch.no_grad():
                for r in records[:8]:
                    gate(); torch.cuda.synchronize(); tick = time.monotonic()
                    prediction = trainer.model([mapper(r)])
                    torch.cuda.synchronize(); relinquish(time.monotonic()-tick)
                    del prediction
            report = {'status': 'passed', 'amp_iterations': 20, 'head_weights_updated': True,
                'batch_size': 2, 'train_image_size': 1024,
                'eval_image_ids': [int(r['image_id']) for r in records[:8]],
                'peak_allocated_mb': torch.cuda.max_memory_allocated()/1024**2,
                'peak_reserved_mb': torch.cuda.max_memory_reserved()/1024**2,
                'seconds': time.monotonic()-started, 'duty_cycle': DUTY,
                'validation_accuracy_evaluated': False, 'test_split_access': False}
            save(report, args.output_dir / 'shared_profile_report.json')
        gate(force=True)
    except Pressure:
        # If pressure was detected before the next update, do not label that
        # not-yet-computed iteration as completed in the checkpoint.
        if args.mode == 'train':
            trainer.iter = last_completed[0]
        trainer.checkpointer.save('resource_stop', iteration=trainer.iter)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['cpu', 'profile', 'train'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu-witness', type=Path)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(str(ROOT))
    if not args.worker:
        launch(args); return
    folder = args.output_dir.resolve()
    code, error = 0, None
    try:
        worker(args)
    except BaseException as exc:
        code, error = (75 if isinstance(exc, Pressure) else 1), repr(exc)
        raise
    finally:
        save({'exit_code': code, 'error': error, 'mode': args.mode,
              'test_split_access': False}, folder / 'completion.json')


if __name__ == '__main__':
    main()
