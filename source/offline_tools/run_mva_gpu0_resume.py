#!/usr/bin/env python3
"""Explicit seed3407 state resume; dynamic cooperative sharing, no external-task mutation."""
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
import run_mva_gpu0_exclusive as previous
from mva_resume_state import file_sha

native = previous.native
UUID = previous.UUID
SOURCE = ROOT / 'output_offline/R01_MVA_MaskRCNN_seed3407_20261022_gpu0_fresh_v1/stage/resource_stop.pth'
SOURCE_SHA = 'a416bf21c09476a414733a656b5a8477b1c7d874602eeec2b26d67bf2f21b00c'
SOURCE_ITERATION = 45720
ALLOCATOR_MB, CONTEXT_MB, RESERVE_MB = 6144, 2048, 4096
TRACKED = ['offline_tools/run_mva_gpu0_resume.py', 'offline_tools/mva_resume_state.py',
           'offline_tools/test_mva_gpu0_resume.py', '.aris/compute/mva-maskrcnn-gpu0-resume.md']


def sharing_duty(gpu, own_pid):
    if gpu['free_mb'] < RESERVE_MB:
        raise native.Pressure('GPU0 free memory below4096MiB')
    return .20 if set(gpu['pids']) - {own_pid} else 1.0


def source_hashes():
    hashes = previous.source_hashes()
    hashes.update({p: native.sha(ROOT / p) for p in TRACKED})
    if file_sha(SOURCE) != SOURCE_SHA:
        raise RuntimeError('Pinned source checkpoint changed')
    hashes['source_checkpoint'] = SOURCE_SHA
    return hashes


def check_evidence(folder, mode, hashes):
    folder = folder.resolve(); folder.relative_to(ROOT / 'output_offline')
    deployment = json.loads((folder / 'deployment.json').read_text())
    completion = json.loads((folder / 'completion.json').read_text())
    report = json.loads((folder / (mode + '_report.json')).read_text())
    if (deployment['mode'] != mode or deployment['source_sha256'] != hashes
            or completion['exit_code'] != 0 or report['status'] != 'passed'):
        raise RuntimeError('Evidence absent/failed/source mismatch')


def duplicate_training():
    rows = native.output(['ps', '-eo', 'pid,comm,args']).splitlines()[1:]
    for row in rows:
        parts = row.split(None, 2)
        if (len(parts) == 3 and parts[1].startswith('python') and '--mode train' in parts[2]
                and '--worker' in parts[2] and int(parts[0]) != os.getpid()):
            if 'run_mva_gpu0_resume.py' in parts[2] or 'run_mva_gpu0_exclusive.py' in parts[2]:
                return True
    return False


def launch(args):
    folder = args.output_dir.resolve(); folder.relative_to(ROOT / 'output_offline')
    hashes = source_hashes()
    gpu = previous.inventory()
    if not args.launch:
        print(json.dumps({'read_only': True, 'gpu': gpu, 'source_checkpoint': str(SOURCE),
                          'source_sha256': hashes, 'output_exists': folder.exists()}))
        return
    if folder.exists():
        raise RuntimeError('Output already exists; no overwrite/duplicate retry')
    if shutil.disk_usage(str(ROOT)).free < 30 * 1024**3:
        raise RuntimeError('Disk headroom below30GiB')
    if duplicate_training():
        raise RuntimeError('Existing own GPU0 training detected; refusing duplicate')
    if args.mode != 'audit':
        if gpu['free_mb'] < ALLOCATOR_MB + CONTEXT_MB + RESERVE_MB:
            raise RuntimeError('Insufficient starting headroom; no retry/queue')
        if args.audit_dir is None:
            raise RuntimeError('Matched CPU audit required')
        check_evidence(args.audit_dir, 'audit', hashes)
    if args.mode == 'train':
        if args.profile_dir is None:
            raise RuntimeError('Matched actual resumed CUDA profile required')
        check_evidence(args.profile_dir, 'profile', hashes)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {'mode': args.mode, 'seed': 3407, 'source_sha256': hashes,
                'source_checkpoint': str(SOURCE), 'source_iteration': SOURCE_ITERATION,
                'start_iter': SOURCE_ITERATION + 1, 'gpu_uuid': UUID,
                'gpu_before': gpu, 'started_unix': time.time(), 'test_split_access': False,
                'rng_state_restored': False, 'dataloader_position_restored': False}
    native.save(manifest, folder / 'deployment.json')
    env = native.native_env('cpu' if args.mode == 'audit' else 'train', folder)
    env['CUDA_VISIBLE_DEVICES'] = '' if args.mode == 'audit' else UUID
    cpus = sorted(os.sched_getaffinity(0))[:4]
    command = ['ionice', '-c', '2', '-n', '7', 'nice', '-n', '10', 'taskset',
               '-c', ','.join(map(str, cpus)), str(native.ENV / 'bin/python'), '-u',
               str(Path(__file__).resolve()), '--worker', '--mode', args.mode,
               '--output-dir', str(folder)]
    with (folder / 'console.log').open('x') as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    manifest.update(pid=process.pid, command=command, cpu_affinity=cpus)
    native.save(manifest, folder / 'deployment.json')
    print(json.dumps({'pid': process.pid, 'folder': str(folder)}), flush=True)


def worker(args):
    folder = args.output_dir.resolve()
    deployment = json.loads((folder / 'deployment.json').read_text())
    if Path(sys.prefix).resolve() != native.ENV.resolve() or deployment['source_sha256'] != source_hashes():
        raise RuntimeError('Runtime/source mismatch')
    if deployment['mode'] != args.mode:
        raise RuntimeError('Mode differs from launch manifest')
    from mva_resume_state import load_checked, restore_full_state
    if args.mode == 'audit':
        for test in ['offline_tools/test_mva_training_state.py', 'offline_tools/test_mva_gpu0_resume.py']:
            subprocess.check_call([sys.executable, '-u', test])
        import torch
        checkpoint, summary = load_checked(SOURCE, torch, SOURCE_SHA, SOURCE_ITERATION)
        native.save({'status': 'passed', 'checkpoint': summary, 'source_checkpoint_sha256': SOURCE_SHA,
                     'native_prefix': sys.prefix, 'source_sha256': deployment['source_sha256'],
                     'cuda_tested': False, 'test_split_access': False}, folder / 'audit_report.json')
        return

    import train_mva_maskrcnn as base
    from mva_training_state import InnerTrainerCheckpointMixin, checkpoint_summary
    torch = base.torch
    torch.set_num_threads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != UUID:
        raise RuntimeError('Wrong GPU binding')
    torch.cuda.set_per_process_memory_fraction(
        ALLOCATOR_MB * 1024**2 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats()
    stage = folder / 'stage'; stage.mkdir(exist_ok=False)
    args.output_dir = stage
    args.seed, args.smoke_only, args.smoke_device = 3407, False, 'cpu'
    cfg = base.make_cfg(args)
    import yaml
    original = yaml.safe_load((SOURCE.parent / 'config.yaml').read_text())
    proposed = yaml.safe_load(cfg.dump())
    original.pop('OUTPUT_DIR'); proposed.pop('OUTPUT_DIR')
    if proposed != original:
        raise RuntimeError('Resume would change frozen scientific config')
    base.default_setup(cfg, args)
    for path, expected in base.INPUTS.items():
        if base.sha256(ROOT / path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    if base.torchvision.__version__ != '0.10.0+cu111':
        raise RuntimeError('Wrong torchvision')
    Path(base.torchvision.__file__).resolve().relative_to(base.OVERLAY)
    base.register_data()
    native.save(dict(deployment, input_sha256=base.INPUTS, frozen_config_equal_except_output=True),
                stage / 'launch_manifest.json')
    if args.mode == 'profile':
        cfg = cfg.clone(); cfg.defrost()
        cfg.DATASETS.TEST = (); cfg.DATALOADER.NUM_WORKERS = 0
        cfg.freeze()

    stopped, last_gate, duty = [None], [0.0], [1.0]
    peers = [None]
    signal.signal(signal.SIGUSR1, lambda *_: stopped.__setitem__(0, 'Requested checkpoint stop'))

    def gate(force=False):
        if not force and time.monotonic() - last_gate[0] < 1.0:
            return
        gpu = previous.inventory()
        import psutil
        own = psutil.Process()
        rss = 0
        for process in [own] + own.children(recursive=True):
            try:
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        if gpu['free_mb'] < RESERVE_MB or rss > 16 * 1024**3:
            native.save(dict(gpu, rss_bytes=rss, our_pid=os.getpid(), timestamp_unix=time.time()),
                        folder / 'resource_event.json')
            raise native.Pressure('GPU0 free-memory/RSS guard; peer appearance alone is allowed')
        new_peers = sorted(set(gpu['pids']) - {os.getpid()})
        duty[0] = sharing_duty(gpu, os.getpid())
        if new_peers != peers[0]:
            native.save({'peer_pids': new_peers, 'duty_cycle': duty[0],
                         'gpu': gpu, 'timestamp_unix': time.time()}, folder / 'sharing_status.json')
            peers[0] = new_peers
        if stopped[0]:
            raise native.Pressure(stopped[0])
        last_gate[0] = time.monotonic()

    def relinquish(elapsed):
        remaining = elapsed * (1.0 / duty[0] - 1.0)
        while remaining > 0:
            tick = min(1.0, remaining); time.sleep(tick); remaining -= tick
            try:
                gate()
            except native.Pressure as exc:
                stopped[0] = str(exc)
                break

    class Trainer(InnerTrainerCheckpointMixin, base.ReplicationTrainer):
        def run_step(self):
            gate(); torch.cuda.synchronize(); tick = time.monotonic()
            super().run_step()
            torch.cuda.synchronize(); relinquish(time.monotonic()-tick)

        @classmethod
        def test(cls, test_cfg, model, evaluators=None):
            class Guarded(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.original = model
                def forward(self, inputs):
                    gate(); torch.cuda.synchronize(); tick = time.monotonic()
                    result = self.original(inputs)
                    torch.cuda.synchronize(); relinquish(time.monotonic()-tick)
                    return result
            return super().test(test_cfg, Guarded(), evaluators)

    gate(force=True)
    trainer = Trainer(cfg)
    report = restore_full_state(trainer, SOURCE, torch, SOURCE_SHA, SOURCE_ITERATION)
    native.save(report, folder / 'resume_report.json')
    completed = [SOURCE_ITERATION]
    if args.mode == 'profile':
        trainer.max_iter = trainer.start_iter + 20
    from detectron2.engine.train_loop import HookBase
    from detectron2.engine import hooks
    import weakref

    class Maintenance(HookBase):
        def after_step(self):
            completed[0] = self.trainer.iter
            if (self.trainer.iter + 1) % 1000 == 0 or stopped[0]:
                self.trainer.checkpointer.save('runtime_latest', iteration=self.trainer.iter)
            if (self.trainer.iter + 1) % 100 == 0:
                native.save({'iteration': self.trainer.iter, 'max_iter': self.trainer.max_iter,
                    'duty_cycle': duty[0], 'peer_pids': peers[0], 'timestamp_unix': time.time()},
                    folder / 'training_status.json')
            if stopped[0]:
                raise native.Pressure(stopped[0])

    maintenance = Maintenance(); maintenance.trainer = weakref.proxy(trainer)
    index = next(i for i,h in enumerate(trainer._hooks) if isinstance(h, hooks.LRScheduler))
    trainer._hooks.insert(index+1, maintenance)
    initial = trainer.model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    try:
        trainer.train()
        if args.mode == 'profile':
            if torch.equal(initial, trainer.model.roi_heads.box_predictor.cls_score.weight):
                raise RuntimeError('Resumed AMP steps did not update head')
            saved, summary = load_checked(stage/'model_final.pth', torch,
                                          expected_iteration=SOURCE_ITERATION+20)
            native.save({'status': 'passed', 'resume': report, 'completed_resumed_steps': 20,
                'head_weights_updated': True, 'new_checkpoint': summary,
                'test_split_access': False, 'validation_accuracy_evaluated': False,
                'peak_reserved_mb': torch.cuda.max_memory_reserved()/1024**2}, folder/'profile_report.json')
        gate(force=True)
    except native.Pressure:
        trainer.iter = completed[0]
        trainer.checkpointer.save('resource_stop', iteration=trainer.iter)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['audit', 'profile', 'train'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); os.chdir(str(ROOT))
    if not args.worker:
        launch(args); return
    folder = args.output_dir.resolve()
    code, error = 0, None
    try:
        worker(args)
    except BaseException as exc:
        code, error = (75 if isinstance(exc, native.Pressure) else 1), repr(exc)
        raise
    finally:
        native.save({'exit_code': code, 'error': error, 'mode': args.mode,
                     'finished_unix': time.time(), 'test_split_access': False}, folder/'completion.json')
        (folder/'exit_code.txt').write_text(str(code)+'\n')


if __name__ == '__main__':
    main()
