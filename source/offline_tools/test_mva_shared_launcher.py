"""Host-only synthetic shared-launcher checks; never invoke Docker or GPU."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_mva_maskrcnn_shared as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='shared_launcher_test_',
                                                dir=str(launcher.ROOT / 'output_offline'))
        self.root = Path(self.temp.name)
        self.profile = self.root / 'output_offline/profile'
        self.profile.mkdir(parents=True)
        self.binary = self.root / '.runtime/mva_tv0100cu111/torchvision/_C.so'
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b'fixture')
        self.source = self.root / 'offline_tools/fixture.py'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('fixture')
        self.report = {'status': 'passed', 'amp_training_steps': 20,
                       'train_image_size': 1024, 'batch_size': 2,
                       'policy': asdict(launcher.SharingPolicy())}
        manifest = {'gpu_uuid': launcher.UUID,
                    'source_sha256': {'offline_tools/fixture.py': launcher.sha256(self.source)},
                    'runtime': {'binary_sha256': launcher.sha256(self.binary)}}
        (self.profile / 'shared_profile_report.json').write_text(json.dumps(self.report))
        (self.profile / 'launch_manifest.json').write_text(json.dumps(manifest))
        (self.profile / 'exit_code.txt').write_text('0\n')
        self.output = self.root / 'output_offline/formal'

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, free=9000, launch=False, detach=False, gpu_processes=''):
        def fake_output(command):
            if command[:2] == ['docker', 'inspect']:
                return launcher.IMAGE
            if command[0] == 'docker':
                return launcher.UUID
            if any(x.startswith('--query-compute-apps=') for x in command):
                return gpu_processes
            return str(free)
        argv = ['launcher', '--profile-dir', str(self.profile),
                '--output-root', str(self.output)]
        argv += ['--launch'] if launch else []
        argv += ['--detach'] if detach else []
        with mock.patch.object(launcher, 'ROOT', self.root), \
             mock.patch.object(launcher.os, 'chdir'), \
             mock.patch.object(launcher, 'command_output', side_effect=fake_output), \
             mock.patch.object(launcher.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*1024**3)), \
             mock.patch.object(launcher.subprocess, 'Popen', return_value=SimpleNamespace(pid=12345)) as popen, \
             mock.patch.object(sys, 'argv', argv), \
             contextlib.redirect_stdout(io.StringIO()) as text:
            launcher.main()
            return text.getvalue(), popen.call_args

    def test_ready_dry_run_never_starts(self):
        text, call = self.invoke()
        self.assertEqual(json.loads(text)['status'], 'shared_ready')
        self.assertIsNone(call)
        self.assertFalse(self.output.exists())

    def test_low_memory_dry_run_is_blocked(self):
        text, call = self.invoke(free=7555)
        self.assertEqual(json.loads(text)['status'], 'blocked_memory_reserve')
        self.assertIsNone(call)

    def test_low_memory_launch_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'Insufficient reserve'):
            self.invoke(free=7555, launch=True)
        self.assertFalse(self.output.exists())

    def test_changed_source_refused(self):
        self.source.write_text('changed')
        with self.assertRaisesRegex(AssertionError, 'Source changed'):
            self.invoke()

    def test_old_profile_policy_refused(self):
        self.report['policy']['context_allowance_mb'] = 512
        (self.profile / 'shared_profile_report.json').write_text(json.dumps(self.report))
        with self.assertRaises(AssertionError):
            self.invoke()

    def test_existing_output_refused(self):
        self.output.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'exists'):
            self.invoke()

    def test_detached_process_is_session_independent(self):
        text, call = self.invoke(launch=True, detach=True)
        self.assertTrue(call.kwargs['start_new_session'])
        self.assertEqual(call.kwargs['stdin'], launcher.subprocess.DEVNULL)
        self.assertNotIn('--detach', call.args[0])
        self.assertIn('12345', text)
        self.assertFalse(self.output.exists())

    def test_existing_log_never_overwritten(self):
        self.output.with_name(self.output.name + '_launcher.log').write_text('existing')
        with self.assertRaises(FileExistsError):
            self.invoke(launch=True, detach=True)

    def test_existing_budget_guard_blocks_dry_run(self):
        text, call = self.invoke(gpu_processes='29058, /workspace/RSPrompter/.venv/bin/python, 15581')
        self.assertEqual(json.loads(text)['status'], 'blocked_existing_budget_guard')
        self.assertIsNone(call)

    def test_existing_budget_guard_blocks_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'board-budget guard'):
            self.invoke(launch=True, gpu_processes='29058, /workspace/RSPrompter/.venv/bin/python, 15581')
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
