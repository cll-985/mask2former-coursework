#!/usr/bin/env python3
"""Opt-in cooperative sharing, explicitly authorized 2026-10-16. GPU0 only."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'offline_tools'))
from mva_shared_gpu_policy import SharingPolicy

GPU_UUID = 'GPU-a332beaa-9bf1-c84b-f27c-8bdcaaebed4c'
PROTOCOL = 'refine-logs/MVA_GPU0_SHARING_PROTOCOL_20261016.md'


def free_memory_mb():
    return int(subprocess.check_output([
        'nvidia-smi', '--id=' + GPU_UUID, '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], universal_newlines=True).strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 3407), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise RuntimeError('Shared runner requires explicit physical GPU0 UUID binding')
    policy = SharingPolicy()
    initial_free = free_memory_mb()
    if not policy.start_allowed(initial_free):
        raise RuntimeError('Need {} MiB free including reserve; found {}'.format(
            policy.start_free_mb, initial_free))
    output = args.output_dir.resolve()
    output.relative_to(ROOT / 'output_offline')
    output.mkdir(parents=True, exist_ok=False)

    # Import uses the already validated private torchvision overlay.
    import train_mva_maskrcnn as base
    torch = base.torch
    if base.torchvision.__version__ != '0.10.0+cu111':
        raise RuntimeError('Unexpected torchvision overlay version')
    Path(base.torchvision.__file__).resolve().relative_to(base.OVERLAY)
    runtime = json.loads((base.OVERLAY / 'runtime_probe.json').read_text())
    if runtime['binary_sha256'] != base.sha256(base.OVERLAY / 'torchvision/_C.so'):
        raise RuntimeError('Private runtime binary changed')
    torch.set_num_threads(2)
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        float(policy.memory_limit_mb * 1024**2 / total_bytes), 0)
    torch.cuda.reset_peak_memory_stats()
    args.smoke_only = False
    args.smoke_device = 'cuda'
    args.output_dir = output
    cfg = base.make_cfg(args)
    if args.profile_only:
        cfg = cfg.clone()
        cfg.defrost()
        cfg.DATASETS.TEST = ()  # Profile manually below; no AP/labels used for selection.
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.freeze()
    base.default_setup(cfg, args)
    for path, expected in base.INPUTS.items():
        if base.sha256(ROOT / path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    base.register_data()
    tracked = [base.CONFIG, 'offline_tools/train_mva_maskrcnn.py',
               'offline_tools/train_mva_maskrcnn_shared.py',
               'offline_tools/mva_shared_gpu_policy.py', PROTOCOL,
               'refine-logs/MVA_ARCH_REPLICATION_PROTOCOL_20261016.md',
               '.aris/compute/remote-maskrcnn.md']
    manifest = {'seed': args.seed, 'gpu_uuid': GPU_UUID, 'policy': asdict(policy),
                'mode': 'shared_amp_profile' if args.profile_only else 'shared_formal_training',
                'source_sha256': {p: base.sha256(ROOT / p) for p in tracked},
                'input_sha256': base.INPUTS, 'pid': os.getpid(),
                'runtime': runtime,
                'test_split_access': False, 'initial_free_mb': initial_free,
                'checkpoint_selection': 'final', 'hardware_compute_isolation': False}
    base.save_json(manifest, output / 'launch_manifest.json')

    class ResourcePressure(RuntimeError):
        pass

    class SharedTrainer(base.ReplicationTrainer):
        def gate(self):
            available = free_memory_mb()
            if not policy.step_allowed(available):
                self.checkpointer.save('resource_stop', iteration=self.iter)
                raise ResourcePressure('Remaining GPU memory {} MiB below reserve'.format(available))

        def relinquish(self, elapsed):
            delay = policy.delay(elapsed)
            # Chunking permits memory checks without a long uninterruptible sleep.
            remaining = delay
            while remaining > 0:
                chunk = min(remaining, 5.0)
                time.sleep(chunk)
                remaining -= chunk
                self.gate()
            return delay

        def run_step(self):
            self.gate()
            torch.cuda.synchronize()
            started = time.monotonic()
            super().run_step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            delay = self.relinquish(elapsed)
            if self.iter < 20 or (self.iter + 1) % 20 == 0:
                row = {'iteration': self.iter, 'active_seconds': elapsed,
                       'yield_seconds': delay, 'free_gpu_mb': free_memory_mb(),
                       'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                       'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                       'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                       'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2}
                base.save_json(row, output / 'sharing_status.json')
                print('SHARED_STEP ' + json.dumps(row), flush=True)

        @classmethod
        def test(cls, test_cfg, model, evaluators=None):
            # Throttle final validation as well as training. Predictions unchanged.
            class CooperativeEvalModel(torch.nn.Module):
                def __init__(self, original):
                    super().__init__()
                    self.original = original

                def forward(self, inputs):
                    if not policy.step_allowed(free_memory_mb()):
                        raise ResourcePressure('GPU pressure during evaluation')
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    result = self.original(inputs)
                    torch.cuda.synchronize()
                    time.sleep(policy.delay(time.monotonic() - started))
                    return result
            return super().test(test_cfg, CooperativeEvalModel(model), evaluators)

    trainer = None
    started = time.monotonic()
    try:
        trainer = SharedTrainer(cfg)
        base.load_verified_pretrain(trainer.model)
        if args.profile_only:
            with base.EventStorage(0) as storage:
                for iteration in range(20):
                    trainer.iter = iteration
                    trainer.run_step()
                    trainer.scheduler.step()
                    storage.step()
            # Stress the largest val images by annotation dimensions, not AP.
            records = base.DatasetCatalog.get('mva_maskrcnn_cardd_val')
            records.sort(key=lambda x: (-x['width'] * x['height'], x['image_id']))
            eval_mapper = base.DatasetMapper(cfg, False)
            trainer.model.eval()
            image_ids = []
            with torch.no_grad():
                for record in records[:8]:
                    trainer.gate()
                    image = eval_mapper(record)
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    prediction = trainer.model([image])[0]['instances']
                    torch.cuda.synchronize()
                    trainer.relinquish(time.monotonic() - tick)
                    image_ids.append(int(record['image_id']))
                    del prediction, image
            report = {'status': 'passed', 'amp_training_steps': 20, 'batch_size': 2,
                      'train_image_size': 1024, 'eval_image_ids': image_ids,
                      'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                      'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
                      'remaining_free_mb': free_memory_mb(), 'policy': asdict(policy),
                      'seconds': time.monotonic() - started,
                      'validation_accuracy_evaluated': False, 'test_split_access': False}
            base.save_json(report, output / 'shared_profile_report.json')
            print('SHARED_PROFILE_PASS ' + json.dumps(report), flush=True)
        else:
            trainer.train()
        (output / 'exit_code.txt').write_text('0\n')
    except BaseException as exc:
        code = 75 if isinstance(exc, ResourcePressure) else 1
        base.save_json({'status': 'stopped', 'error': repr(exc), 'exit_code': code,
                        'test_split_access': False}, output / 'stopped_report.json')
        (output / 'exit_code.txt').write_text(str(code) + '\n')
        raise


if __name__ == '__main__':
    main()
