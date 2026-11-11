#!/usr/bin/env python3
"""Copy only our validated conda runtime; no installs/daemon changes/overwrite."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / '.runtime_gpu2_native_v2'
IMAGE = 'sha256:c35f519fe286f089f9728277fadcd8991ead8c80038cf0026f851292bdd7b97a'


def main():
    os.chdir(str(ROOT))
    image = subprocess.check_output(['docker', 'inspect', '--format', '{{.Image}}',
                                     'm2f_container'], universal_newlines=True).strip()
    if image != IMAGE:
        raise RuntimeError('Source image changed')
    DEST.mkdir(exist_ok=False)
    if '--worker' not in sys.argv:
        with (DEST / 'copy.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__, '--worker-existing'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(json.dumps({'copy_pid': child.pid, 'destination': str(DEST)}))


def worker():
    source = subprocess.Popen(['docker', 'exec', 'm2f_container', 'nice', '-n', '10',
                               'tar', '-C', '/opt', '-cf', '-', 'conda'],
                              stdout=subprocess.PIPE)
    target = subprocess.Popen(['ionice', '-c', '3', 'nice', '-n', '10', 'tar',
                               '-xf', '-', '-C', str(DEST), '--no-same-owner'],
                              stdin=source.stdout)
    source.stdout.close()
    target_code = target.wait()
    source_code = source.wait()
    result = {'source_exit': source_code, 'extract_exit': target_code,
              'source_image': IMAGE, 'source': 'm2f_container:/opt/conda',
              'destination': str(DEST / 'conda')}
    (DEST / 'copy_status.json').write_text(json.dumps(result, indent=2) + '\n')
    if source_code or target_code:
        raise RuntimeError('Copy failed; preserve directory, no retry/overwrite')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    if '--worker-existing' in sys.argv:
        worker()
    else:
        main()
