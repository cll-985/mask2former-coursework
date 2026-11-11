#!/usr/bin/env python3
"""Install one hash-pinned offline wheel into a private project overlay only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
WHEEL = ROOT / 'artifacts/mva_20261016/runtime/torchvision-0.10.0+cu111-cp37-cp37m-linux_x86_64.whl'
EXPECTED = 'c8c811869d3afeb6082220cd7a136b65822bbc86cfbeeb02c4624bd262fe06ce'
TARGET = ROOT / '.runtime/mva_tv0100cu111'


def main():
    os.chdir(str(ROOT))
    assert hashlib.sha256(WHEEL.read_bytes()).hexdigest() == EXPECTED
    if not TARGET.exists():
        temporary = ROOT / '.runtime/pip_tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TMPDIR=str(temporary), PIP_NO_CACHE_DIR='1')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index',
                               '--no-deps', '--no-compile', '--no-cache-dir',
                               '--disable-pip-version-check', '--target', str(TARGET),
                               str(WHEEL)], env=env)
    sys.path.insert(0, str(TARGET))
    import torch
    import torchvision
    assert torch.__version__ == '1.9.0' and torch.version.cuda == '11.1'
    assert torchvision.__version__ == '0.10.0+cu111'
    Path(torchvision.__file__).resolve().relative_to(TARGET)
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]])
    kept = torchvision.ops.nms(boxes, torch.tensor([.9, .8]), .5).tolist()
    assert kept == [0]
    features = torch.arange(16., requires_grad=True).reshape(1, 1, 4, 4)
    aligned = torchvision.ops.roi_align(features, [boxes[:1]], output_size=2)
    gradient = torch.autograd.grad(aligned.sum(), features)[0]
    assert torch.isfinite(gradient).all().item()
    result = {'status': 'passed', 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'wheel_sha256': EXPECTED,
              'overlay': str(TARGET), 'cpu_nms_indices': kept,
              'roi_align_backward_finite': True, 'base_packages_changed': False,
              'binary_sha256': hashlib.sha256((TARGET / 'torchvision/_C.so').read_bytes()).hexdigest()}
    (TARGET / 'runtime_probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
