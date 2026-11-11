#!/usr/bin/env python3
"""Fresh seed3407 on physical GPU0; optional explicitly authorized bounded sharing."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'offline_tools'))
import run_mva_gpu2_native as native

UUID = 'GPU-a332beaa-9bf1-c84b-f27c-8bdcaaebed4c'
TRACKED = native.TRACKED + [
    'offline_tools/run_mva_gpu0_exclusive.py',
    'offline_tools/test_mva_training_state.py',
    '.aris/compute/mva-maskrcnn-gpu0-exclusive.md',
    '.aris/compute/mva-maskrcnn-gpu0-shared.md']
ALLOCATOR_MB, RESERVE_MB = 16384, 4096


def placement(args):
    external = getattr(args, 'share_with_pid', 0)
    if external < 0:
        raise ValueError('Expected positive external PID or zero for exclusive mode')
    return {'shared_with_pid': external, 'allocator_mb': 6144 if external else ALLOCATOR_MB,
            'duty_cycle': 0.20 if external else 1.0}


def check_placement(gpu, policy):
    external = policy['shared_with_pid']
    if external:
        if set(gpu['pids']) != {external} or gpu['free_mb'] < 6144 + 2048 + RESERVE_MB:
            raise RuntimeError('Shared PID differs/extra task/insufficient headroom')
    elif gpu['pids'] or gpu['used_mb'] >= 500 or gpu['utilization'] > 10:
        raise RuntimeError('Physical GPU0 not idle; no waiting/retry')


def inventory():
    row = native.output(['nvidia-smi', '-i', UUID,
        '--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits']).split(',')
    if row[0].strip() != UUID:
        raise RuntimeError('GPU identity mismatch')
    lines = native.output(['nvidia-smi', '-i', UUID,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    return {'uuid': UUID, 'used_mb': int(row[1]), 'free_mb': int(row[2]),
            'utilization': int(row[3]),
            'pids': sorted(int(p.strip()) for p in lines.splitlines() if p.strip())}


def source_hashes():
    native.verify_sources()
    result = {p: native.sha(ROOT / p) for p in TRACKED}
    result['native_python'] = native.sha(native.ENV / 'bin/python')
    result['torchvision_binary'] = native.sha(ROOT / '.runtime/mva_tv0100cu111/torchvision/_C.so')
    return result


def evidence(folder, mode, hashes, policy):
    folder = folder.resolve()
    folder.relative_to(ROOT / 'output_offline')
    state = json.loads((folder / 'deployment.json').read_text())
    end = json.loads((folder / 'completion.json').read_text())
    name = 'smoke_report.json' if mode == 'cpu' else 'profile_report.json'
    report = json.loads((folder / 'stage' / name).read_text())
    if (state['mode'] != mode or state['source_sha256'] != hashes
            or end['exit_code'] != 0 or report['status'] != 'passed'):
        raise RuntimeError('Witness failed or source/runtime changed')
    if mode == 'profile' and state['placement'] != policy:
        raise RuntimeError('Profile sharing placement differs from production')
    if mode == 'profile' and policy['shared_with_pid']:
        witnessed = json.loads((folder / 'placement.json').read_text())
        current = native.output(['ps', '-p', str(policy['shared_with_pid']), '-o', 'lstart='])
        if current != witnessed['process_start']:
            raise RuntimeError('External PID was reused after profile')


def launch(args):
    folder = args.output_dir.resolve()
    folder.relative_to(ROOT / 'output_offline')
    hashes = source_hashes()
    gpu = inventory()
    policy = placement(args)
    if not args.launch:
        print(json.dumps({'gpu': gpu, 'mode': args.mode,
                          'output_exists': folder.exists(), 'read_only': True}))
        return
    if folder.exists():
        raise RuntimeError('Output exists; refusing overwrite/duplicate launch')
    copy = json.loads((native.ENV.parent / 'copy_status.json').read_text())
    if copy['source_exit'] or copy['extract_exit']:
        raise RuntimeError('Runtime copy incomplete')
    if shutil.disk_usage(str(ROOT)).free < 30 * 1024**3:
        raise RuntimeError('Insufficient disk headroom')
    if args.mode != 'cpu':
        check_placement(gpu, policy)
        if args.cpu_witness is None:
            raise RuntimeError('CPU witness required')
        evidence(args.cpu_witness, 'cpu', hashes, policy)
    if args.mode == 'train':
        if args.profile_dir is None:
            raise RuntimeError('CUDA AMP profile required')
        evidence(args.profile_dir, 'profile', hashes, policy)
    folder.mkdir(parents=True, exist_ok=False)
    state = {'mode': args.mode, 'seed': 3407, 'gpu_uuid': UUID,
             'source_sha256': hashes, 'gpu_before': gpu,
             'placement': policy,
             'fresh_initialization': True, 'legacy_checkpoint_loaded': False,
             'test_split_access': False, 'started_unix': time.time()}
    native.save(state, folder / 'deployment.json')
    env = native.native_env(args.mode, folder)
    env['CUDA_VISIBLE_DEVICES'] = '' if args.mode == 'cpu' else UUID
    cpus = sorted(os.sched_getaffinity(0))[:4]
    cmd = ['ionice', '-c', '2', '-n', '7', 'nice', '-n', '10',
           'taskset', '-c', ','.join(map(str, cpus)), str(native.ENV / 'bin/python'),
           '-u', str(Path(__file__).resolve()), '--worker', '--mode', args.mode,
           '--output-dir', str(folder)]
    if policy['shared_with_pid']:
        cmd += ['--share-with-pid', str(policy['shared_with_pid'])]
    with (folder / 'console.log').open('x') as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state.update(pid=proc.pid, command=cmd, cpu_affinity=cpus)
    native.save(state, folder / 'deployment.json')
    print(json.dumps({'pid': proc.pid, 'folder': str(folder)}), flush=True)


def worker(args):
    folder = args.output_dir.resolve()
    state = json.loads((folder / 'deployment.json').read_text())
    policy = placement(args)
    if state['placement'] != policy:
        raise RuntimeError('Placement differs from launch record')
    if Path(sys.prefix).resolve() != native.ENV.resolve():
        raise RuntimeError('Wrong native environment')
    if state['source_sha256'] != source_hashes() or state['mode'] != args.mode:
        raise RuntimeError('Source/runtime/mode changed since launch')
    if args.mode == 'cpu':
        subprocess.check_call([sys.executable, '-u',
                               'offline_tools/test_mva_training_state.py'])
    import train_mva_maskrcnn as base
    from mva_training_state import InnerTrainerCheckpointMixin, checkpoint_summary
    torch = base.torch
    torch.set_num_threads(2)
    args.output_dir = folder / 'stage'
    args.output_dir.mkdir(exist_ok=False)
    args.seed, args.smoke_only, args.smoke_device = 3407, args.mode == 'cpu', 'cpu'
    cfg = base.make_cfg(args)
    base.default_setup(cfg, args)
    for path, expected in base.INPUTS.items():
        if base.sha256(ROOT / path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    if base.torchvision.__version__ != '0.10.0+cu111':
        raise RuntimeError('Wrong torchvision')
    Path(base.torchvision.__file__).resolve().relative_to(base.OVERLAY)
    base.register_data()
    native.save(dict(state, input_sha256=base.INPUTS, native_prefix=sys.prefix),
                args.output_dir / 'launch_manifest.json')
    if args.mode == 'cpu':
        native.save(base.smoke(cfg, args.output_dir), args.output_dir / 'smoke_report.json')
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
        raise RuntimeError('Wrong physical GPU binding')
    torch.cuda.set_per_process_memory_fraction(
        policy['allocator_mb'] * 1024**2 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats()
    stop = [None]
    last_gate = [0.0]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, 'Requested checkpoint stop'))
    external = policy['shared_with_pid']
    external_start = (native.output(['ps', '-p', str(external), '-o', 'lstart='])
                      if external else None)
    native.save({'expected_external_pid': external, 'process_start': external_start,
                 'duty_cycle': policy['duty_cycle'], 'allocator_mb': policy['allocator_mb'],
                 'reserve_mb': RESERVE_MB, 'external_progress_monitored': False},
                folder / 'placement.json')

    def gate(force=False):
        if not force and time.monotonic() - last_gate[0] < 1.0:
            return
        gpu = inventory()
        import psutil
        process = psutil.Process()
        rss = sum(p.memory_info().rss for p in [process] + process.children(recursive=True)
                  if p.is_running())
        expected = {os.getpid()} | ({external} if external else set())
        external_changed = False
        if external:
            try:
                external_changed = native.output(['ps', '-p', str(external), '-o', 'lstart=']) != external_start
            except subprocess.CalledProcessError:
                external_changed = True
        if (set(gpu['pids']) - expected or (external and external not in gpu['pids'])
                or external_changed or gpu['free_mb'] < RESERVE_MB or rss > 16 * 1024**3):
            native.save(dict(gpu, our_pid=os.getpid(), rss_bytes=rss,
                             timestamp_unix=time.time()), folder / 'resource_event.json')
            raise native.Pressure('GPU0 placement/external-PID/memory/RSS guard triggered')
        if stop[0]:
            raise native.Pressure(stop[0])
        last_gate[0] = time.monotonic()

    def relinquish(elapsed):
        remaining = elapsed * (1.0 / policy['duty_cycle'] - 1.0)
        while remaining > 0:
            tick = min(1.0, remaining)
            time.sleep(tick)
            remaining -= tick
            try:
                gate()
            except native.Pressure as exc:
                # Finish scheduler hooks before writing the updated boundary.
                stop[0] = str(exc)
                break

    if args.mode == 'profile':
        cfg = cfg.clone(); cfg.defrost()
        cfg.SOLVER.MAX_ITER = 20
        cfg.SOLVER.CHECKPOINT_PERIOD = 20
        cfg.SOLVER.STEPS = ()
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.freeze()

    class Trainer(InnerTrainerCheckpointMixin, base.ReplicationTrainer):
        def run_step(self):
            gate()
            torch.cuda.synchronize(); tick = time.monotonic()
            super().run_step()
            torch.cuda.synchronize()
            relinquish(time.monotonic() - tick)

        @classmethod
        def test(cls, test_cfg, model, evaluators=None):
            class Guarded(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.original = model
                def forward(self, inputs):
                    gate()
                    torch.cuda.synchronize(); tick = time.monotonic()
                    result = self.original(inputs)
                    torch.cuda.synchronize()
                    relinquish(time.monotonic() - tick)
                    return result
            return super().test(test_cfg, Guarded(), evaluators)

    gate(force=True)
    trainer = Trainer(cfg)
    base.load_verified_pretrain(trainer.model)
    initial = trainer.model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    from detectron2.engine.train_loop import HookBase
    from detectron2.engine import hooks
    completed = [-1]

    class Maintenance(HookBase):
        def after_step(self):
            completed[0] = self.trainer.iter
            if args.mode == 'train' and ((self.trainer.iter + 1) % 1000 == 0 or stop[0]):
                self.trainer.checkpointer.save('runtime_latest', iteration=self.trainer.iter)
            if (self.trainer.iter + 1) % 100 == 0:
                native.save({'iteration': self.trainer.iter, 'max_iter': cfg.SOLVER.MAX_ITER,
                    'gpu_uuid': UUID, 'peak_reserved_mb': torch.cuda.max_memory_reserved()/1024**2,
                    'duty_cycle': policy['duty_cycle'],
                    'timestamp_unix': time.time()}, folder / 'training_status.json')
            if stop[0]:
                raise native.Pressure(stop[0])

    # Scheduler has advanced before any resource-stop boundary is recorded.
    # Insert before periodic checkpoint/eval hooks so final eval pressure cannot
    # falsely label the just-completed update as an earlier iteration.
    scheduler_index = next(i for i,h in enumerate(trainer._hooks) if isinstance(h, hooks.LRScheduler))
    maintenance = Maintenance()
    import weakref
    maintenance.trainer = weakref.proxy(trainer)
    trainer._hooks.insert(scheduler_index + 1, maintenance)
    started = time.monotonic()
    try:
        trainer.train()
        if args.mode == 'profile':
            if torch.equal(initial, trainer.model.roi_heads.box_predictor.cls_score.weight):
                raise RuntimeError('AMP profile did not update the head')
            checkpoint_path = args.output_dir / 'model_final.pth'
            saved = torch.load(str(checkpoint_path), map_location='cpu')
            summary = checkpoint_summary(saved)
            if summary['optimizer_resume_status'] != 'complete':
                raise RuntimeError('Actual AMP checkpoint incomplete: ' + repr(summary))
            scaler = trainer._trainer.grad_scaler
            scaler_state = scaler.state_dict()
            first = next(p for p in trainer.model.parameters() if p in trainer.optimizer.state)
            trainer.optimizer.state[first]['momentum_buffer'].zero_()
            trainer.optimizer.param_groups[0]['lr'] = 0.0
            changed_scaler = dict(scaler_state); changed_scaler['scale'] = 1.0
            scaler.load_state_dict(changed_scaler)
            trainer.checkpointer.load(str(checkpoint_path))
            if scaler.state_dict() != saved['trainer']['_trainer']['grad_scaler']:
                raise RuntimeError('Actual CUDA GradScaler reload differs')
            opt = trainer.optimizer.state_dict()
            expected = saved['trainer']['_trainer']['optimizer']
            if opt['param_groups'] != expected['param_groups']:
                raise RuntimeError('Optimizer groups reload differs')
            for key, item in expected['state'].items():
                if not torch.equal(opt['state'][key]['momentum_buffer'].cpu(), item['momentum_buffer']):
                    raise RuntimeError('Momentum reload differs')
            if trainer.scheduler.state_dict() != saved['trainer']['hooks']['LRScheduler']:
                raise RuntimeError('Scheduler reload differs')
            records = base.DatasetCatalog.get('mva_maskrcnn_cardd_val')
            records.sort(key=lambda r: (-r['width']*r['height'], r['image_id']))
            mapper = base.DatasetMapper(cfg, False)
            trainer.model.eval()
            with torch.no_grad():
                for record in records[:8]:
                    gate(force=True)
                    torch.cuda.synchronize(); tick = time.monotonic()
                    prediction = trainer.model([mapper(record)])[0]['instances']
                    torch.cuda.synchronize()
                    relinquish(time.monotonic()-tick)
                    if not torch.isfinite(prediction.scores).all() or not torch.isfinite(prediction.pred_boxes.tensor).all():
                        raise RuntimeError('Nonfinite validation predictions')
                    del prediction
            native.save({'status': 'passed', 'amp_iterations': 20, 'head_weights_updated': True,
                'checkpoint_state': summary, 'actual_amp_state_reload_verified': True,
                'batch_size': 2, 'image_size': 1024,
                'placement': policy,
                'eval_image_ids': [int(r['image_id']) for r in records[:8]],
                'peak_reserved_mb': torch.cuda.max_memory_reserved()/1024**2,
                'seconds': time.monotonic()-started, 'test_split_access': False,
                'validation_accuracy_evaluated': False}, args.output_dir / 'profile_report.json')
        gate(force=True)
    except native.Pressure:
        if completed[0] >= 0:
            trainer.iter = completed[0]
            trainer.checkpointer.save('resource_stop', iteration=trainer.iter)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['cpu', 'profile', 'train'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu-witness', type=Path)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--share-with-pid', type=int, default=0,
                        help='Explicitly authorized sharing with exactly this existing GPU0 compute PID')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(str(ROOT))
    if not args.worker:
        launch(args)
        return
    folder = args.output_dir.resolve()
    code, error = 0, None
    try:
        worker(args)
    except BaseException as exc:
        code, error = (75 if isinstance(exc, native.Pressure) else 1), repr(exc)
        raise
    finally:
        native.save({'exit_code': code, 'error': error, 'mode': args.mode,
                     'finished_unix': time.time(), 'test_split_access': False},
                    folder / 'completion.json')
        (folder / 'exit_code.txt').write_text(str(code) + '\n')


if __name__ == '__main__':
    main()
