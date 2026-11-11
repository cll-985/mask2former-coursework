#!/usr/bin/env python3
"""Resume the latest verified109275 checkpoint without rewriting historical entrypoints."""
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mva_gpu0_resume as engine

engine.SOURCE = engine.ROOT / 'output_offline/R01_MVA_MaskRCNN_seed3407_20261023_gpu0_resume_v1/stage/resource_stop.pth'
engine.SOURCE_SHA = 'ec5968dbd82252ea220dd62b900acc8cc25f24babbdedf05ca895ddda4fd5202'
engine.SOURCE_ITERATION = 109275
_base_hashes = engine.source_hashes


def source_hashes():
    result = _base_hashes()
    for relative in ['offline_tools/run_mva_gpu0_resume_v2.py',
                     'offline_tools/test_mva_gpu0_resume_v2.py',
                     '.aris/compute/mva-maskrcnn-gpu0-resume-v2.md']:
        result[relative] = engine.native.sha(engine.ROOT / relative)
    return result


def duplicate_training():
    for row in engine.native.output(['ps', '-eo', 'pid,comm,args']).splitlines()[1:]:
        parts = row.split(None, 2)
        if (len(parts) == 3 and parts[1].startswith('python') and '--mode train' in parts[2]
                and '--worker' in parts[2] and int(parts[0]) != os.getpid()
                and ('run_mva_gpu0_resume' in parts[2] or 'run_mva_gpu0_exclusive.py' in parts[2])):
            return True
    return False


engine.source_hashes = source_hashes
engine.duplicate_training = duplicate_training
# The detached child's script must be THIS adapter, not the historical launcher.
engine.__file__ = __file__


if __name__ == '__main__':
    if '--worker' in sys.argv and '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1] == 'audit':
        subprocess.check_call([sys.executable, '-u', str(engine.ROOT / 'offline_tools/test_mva_gpu0_resume_v2.py')])
    engine.main()
