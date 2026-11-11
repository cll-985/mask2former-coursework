#!/usr/bin/env python3
"""Continue the verified321627 GPU1 resource-stop checkpoint on physical GPU0."""
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mva_gpu0_resume as engine

UUID = 'GPU-a332beaa-9bf1-c84b-f27c-8bdcaaebed4c'
engine.UUID = UUID
engine.previous.UUID = UUID
engine.SOURCE = engine.ROOT / 'output_offline/R01_MVA_MaskRCNN_seed3407_20261025_gpu1_resume_v1/stage/resource_stop.pth'
engine.SOURCE_SHA = '587c7776b2c6560f5eb9da9ab6990e1618d6782ad0be675e471adba878a7fa4e'
engine.SOURCE_ITERATION = 321627
_base_hashes = engine.source_hashes


def source_hashes():
    result = _base_hashes()
    for relative in ['offline_tools/run_mva_gpu0_resume_v3.py',
                     'offline_tools/test_mva_gpu0_resume_v3.py',
                     '.aris/compute/mva-maskrcnn-gpu0-resume-v3.md']:
        result[relative] = engine.native.sha(engine.ROOT / relative)
    return result


def duplicate_training():
    for row in engine.native.output(['ps', '-eo', 'pid,comm,args']).splitlines()[1:]:
        parts = row.split(None, 2)
        if (len(parts) == 3 and parts[1].startswith('python')
                and '--mode train' in parts[2] and '--worker' in parts[2]
                and int(parts[0]) != os.getpid()
                and ('run_mva_gpu0_resume' in parts[2]
                     or 'run_mva_gpu0_exclusive.py' in parts[2]
                     or 'run_mva_gpu1_resume' in parts[2])):
            return True
    return False


engine.source_hashes = source_hashes
engine.duplicate_training = duplicate_training
engine.__file__ = __file__


if __name__ == '__main__':
    if '--worker' in sys.argv and '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1] == 'audit':
        subprocess.check_call([sys.executable, '-u', str(engine.ROOT / 'offline_tools/test_mva_gpu0_resume_v3.py')])
    engine.main()
