#!/usr/bin/env python3
"""Stdlib-only launcher safety tests; fake inventories, no GPU/docker execution."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_mva_maskrcnn as launcher


class PreflightTests(unittest.TestCase):
    def setUp(self):
        output = launcher.ROOT / 'output_offline'
        output.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='mva_launcher_test_', dir=str(output))
        self.root = Path(self.temporary.name)
        self.smoke = self.root / 'output_offline/smoke'
        self.smoke.mkdir(parents=True)
        for path in launcher.TRACKED:
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('synthetic_fixture')
        binary = self.root / '.runtime/mva_tv0100cu111/torchvision/_C.so'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'synthetic_binary')
        (self.root / '.aris/compute/mva-maskrcnn-gpu1.md').write_text('gpu1_fixture')
        self.manifest = {'source_sha256': {p: launcher.digest(self.root / p) for p in launcher.TRACKED},
                         'runtime': {'binary_sha256': launcher.digest(binary)}}
        (self.smoke / 'launch_manifest.json').write_text(json.dumps(self.manifest))
        (self.smoke / 'smoke_report.json').write_text(json.dumps(
            {'status': 'passed', 'checkpoint_reload_predictions_exact': True}))
        self.target = self.root / 'output_offline/formal'
        self.args = ['launcher', '--container', 'm2f_gpu3', '--gpu-uuid', 'GPU-fixture',
                     '--smoke-dir', str(self.smoke), '--output-root', str(self.target)]

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, memory=0, utilization=0, exposed=True, launch=False,
               detach=False, container='m2f_gpu3', image=None):
        def fake_run(command):
            if command[:2] == ['docker', 'inspect']:
                return image or (launcher.GPU1_EXISTING_IMAGE if container == 'm2f_container'
                                 else launcher.EXPECTED_IMAGE)
            uuid = 'GPU-fixture' if exposed or command[0] != 'docker' else 'GPU-other'
            return '0, {}, {}, {}'.format(uuid, memory, utilization)
        argv = list(self.args)
        argv[2] = container
        argv += ['--launch'] if launch else []
        argv += ['--detach'] if detach else []
        # Mocked process only; tests never execute Docker or access a live GPU.
        with mock.patch.object(launcher, 'ROOT', self.root), \
             mock.patch.object(launcher, 'run', side_effect=fake_run), \
             mock.patch.object(launcher.os, 'chdir'), \
             mock.patch.object(launcher.subprocess, 'Popen', return_value=mock.Mock(pid=12345)) as popen, \
             mock.patch.object(sys, 'argv', argv), \
             contextlib.redirect_stdout(io.StringIO()) as captured:
            launcher.main()
            self.popen_call = popen.call_args
            return json.JSONDecoder().raw_decode(captured.getvalue())[0]

    def test_idle_default_is_read_only(self):
        self.assertEqual(self.invoke()['status'], 'preflight_ready')
        self.assertFalse(self.target.exists())
        self.assertIsNone(self.popen_call)

    def test_busy_reports_blocked(self):
        self.assertEqual(self.invoke(memory=18000, utilization=90)['status'], 'blocked_gpu_busy')
        self.assertFalse(self.target.exists())

    def test_busy_launch_is_refused_without_output(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU is busy'):
            self.invoke(memory=18000, utilization=90, launch=True)
        self.assertFalse(self.target.exists())

    def test_low_memory_but_active_is_not_idle(self):
        self.assertEqual(self.invoke(memory=100, utilization=90)['status'], 'blocked_gpu_busy')

    def test_unexposed_uuid_is_refused(self):
        with self.assertRaisesRegex(AssertionError, 'not exposed'):
            self.invoke(exposed=False)

    def test_changed_source_is_refused(self):
        (self.root / launcher.TRACKED[0]).write_text('changed')
        with self.assertRaisesRegex(AssertionError, 'Changed since smoke'):
            self.invoke()

    def test_existing_gpu1_container_image_is_pinned(self):
        state = self.invoke(container='m2f_container')
        self.assertEqual(state['container_image'], launcher.GPU1_EXISTING_IMAGE)
        self.assertIn('deployment_ledger_sha256', state)
        self.assertIsNone(self.popen_call)

    def test_wrong_gpu1_image_refused(self):
        with self.assertRaisesRegex(AssertionError, 'image differs'):
            self.invoke(container='m2f_container', image=launcher.EXPECTED_IMAGE)

    def test_detached_session_is_network_independent(self):
        self.invoke(launch=True, detach=True)
        self.assertTrue(self.popen_call.kwargs['start_new_session'])
        self.assertEqual(self.popen_call.kwargs['stdin'], launcher.subprocess.DEVNULL)
        self.assertNotIn('--detach', self.popen_call.args[0])
        self.assertFalse(self.target.exists())

    def test_existing_detached_log_not_overwritten(self):
        self.target.with_name(self.target.name + '_launcher.log').write_text('existing')
        with self.assertRaises(FileExistsError):
            self.invoke(launch=True, detach=True)


if __name__ == '__main__':
    unittest.main()
