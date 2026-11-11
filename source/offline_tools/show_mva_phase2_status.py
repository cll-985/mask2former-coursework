#!/usr/bin/env python3
"""Read phase2 progress on the host or in the container; stdlib only."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='output_offline/MVA_phase2_20261015_v1')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    run = (root / args.run).resolve()
    run.relative_to(root)
    with (run / 'status.json').open() as handle:
        state = json.load(handle)
    print('RUN:', str(run))
    print('STATUS:', state['status'], 'START:', state['started_at'])
    print('CPU workers:', state['cpu_workers'], 'GPU disabled:', state['gpu_visible_devices'] == '')
    for name, task in sorted(state['tasks'].items()):
        print('{}: {} pid={} exit={}'.format(name, task['status'], task.get('pid'), task.get('exit_code', '-')))
    for path in sorted((run / 'controls').glob('*/controls_report.json')):
        with path.open() as handle:
            report = json.load(handle)
        print('CONTROLS {}: {}/{}'.format(path.parent.name, len(report['rules']), len(report['protocol']['rules'])))
    progress = run / 'primary_bootstrap/progress.json'
    if progress.exists():
        with progress.open() as handle:
            values = json.load(handle)
        done, target = values['completed'], values['target']
        seconds = values['elapsed_seconds']
        remaining = (target - done) * seconds / max(done, 1)
        print('BOOTSTRAP: {}/{} ({:.2f}%), elapsed={:.1f} min, estimated remaining={:.1f} min'.format(
            done, target, 100.0 * done / target, seconds / 60, remaining / 60))
        print('Estimate varies with shared CPU load; partial intervals are not stopping criteria.')
    else:
        print('BOOTSTRAP: initializing / fewer than 10 draws; inspect console log if needed.')
    if state['status'] in ('failed', 'running_with_failed_tasks'):
        for name, task in state['tasks'].items():
            if task['status'] == 'failed':
                path = run / (name + '_console.log')
                print('FAILED LOG:', str(path))
                with path.open(errors='replace') as handle:
                    print(''.join(handle.readlines()[-12:]))
    elif state['status'] == 'complete':
        print('ALL_TASKS_COMPLETE; read final reports before updating paper claims.')


if __name__ == '__main__':
    main()
