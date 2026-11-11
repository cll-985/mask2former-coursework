#!/usr/bin/env python3
"""Migrate the scheduler-aligned seed3407 state to idle physical GPU1."""
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mva_gpu0_resume as engine

UUID = 'GPU-cccb4ba2-2fc7-cbfe-1fbf-635df38bdbb3'
SOURCE_ITERATION = 236751
SOURCE_SHA = '3182a3879513ac59e6145271a5fcd846e0b26e5c166b4280ff835e2618e5add5'
engine.SOURCE = engine.ROOT / 'output_offline/R01_MVA_MaskRCNN_seed3407_20261023_gpu0_resume_v2/stage/resource_stop.pth'
engine.SOURCE_SHA = SOURCE_SHA
engine.SOURCE_ITERATION = SOURCE_ITERATION
# Both binding and the inventory function's defining-module global must agree.
engine.UUID = UUID
engine.previous.UUID = UUID
_base_hashes = engine.source_hashes
_base_launch = engine.launch


def source_hashes():
    hashes = _base_hashes()
    for relative in ['offline_tools/run_mva_gpu1_resume_20261025.py',
                     'offline_tools/test_mva_gpu1_resume_20261025.py',
                     '.aris/compute/mva-maskrcnn-gpu1-resume-20261025.md']:
        hashes[relative] = engine.native.sha(engine.ROOT / relative)
    return hashes


def duplicate_training():
    for row in engine.native.output(['ps', '-eo', 'pid,comm,args']).splitlines()[1:]:
        parts = row.split(None, 2)
        if (len(parts) == 3 and parts[1].startswith('python')
                and int(parts[0]) != os.getpid()
                and '--worker' in parts[2] and '--mode train' in parts[2]
                and ('run_mva_gpu0_resume' in parts[2]
                     or 'run_mva_gpu0_exclusive.py' in parts[2]
                     or 'run_mva_gpu1_resume' in parts[2])):
            return True
    return False


def require_idle(gpu):
    if gpu['uuid'] != UUID:
        raise RuntimeError('Physical GPU1 UUID mismatch')
    if gpu['pids'] or gpu['used_mb'] >= 500 or gpu['utilization'] > 10:
        raise RuntimeError('Physical GPU1 is not idle; no queue or forced sharing')


def launch(args):
    if args.launch and args.mode != 'audit':
        require_idle(engine.previous.inventory())
    return _base_launch(args)


engine.source_hashes = source_hashes
engine.duplicate_training = duplicate_training
engine.launch = launch
# Detached children must use this adapter, never an old GPU0 pinned source.
engine.__file__ = __file__


if __name__ == '__main__':
    if '--worker' in sys.argv and '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1] == 'audit':
        subprocess.check_call([sys.executable, '-u', str(engine.ROOT / 'offline_tools/test_mva_gpu1_resume_20261025.py')])
    engine.main()
