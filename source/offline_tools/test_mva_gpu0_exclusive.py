"""GPU0 launch safety tests: synthetic fixtures, no CUDA/GPU/child execution."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('gpu0', Path(__file__).with_name('run_mva_gpu0_exclusive.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Safety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = self.root / 'env/conda'
        (self.env / 'bin').mkdir(parents=True)
        (self.env / 'bin/python').write_text('fixture')
        (self.env.parent / 'copy_status.json').write_text(json.dumps({'source_exit': 0, 'extract_exit': 0}))
        (self.root / 'output_offline').mkdir()
        self.args = argparse.Namespace(mode='train', output_dir=self.root / 'output_offline/new',
            cpu_witness=self.root / 'cpu', profile_dir=self.root / 'profile', launch=True)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(m, 'ROOT', self.root))
        self.stack.enter_context(patch.object(m.native, 'ENV', self.env))
        self.stack.enter_context(patch.object(m, 'source_hashes', return_value={'source': 'hash'}))
        self.gpu = self.stack.enter_context(patch.object(m, 'inventory', return_value={
            'uuid': m.UUID, 'used_mb': 8, 'free_mb': 24260, 'utilization': 0, 'pids': []}))
        self.stack.enter_context(patch.object(m.shutil, 'disk_usage', return_value=type('D', (), {'free': 100 * 1024**3})()))
        self.check = self.stack.enter_context(patch.object(m, 'evidence'))
        self.popen = self.stack.enter_context(patch.object(m.subprocess, 'Popen'))
        self.popen.return_value.pid = 123
        self.stack.enter_context(patch.object(m.os, 'sched_getaffinity', return_value={0, 1, 2, 3}, create=True))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def rejected(self):
        with self.assertRaises((RuntimeError, ValueError)):
            m.launch(self.args)
        self.popen.assert_not_called()

    def test_default_read_only(self):
        self.args.launch = False
        m.launch(self.args)
        self.assertFalse(self.args.output_dir.exists())
        self.popen.assert_not_called()

    def test_existing_output(self):
        self.args.output_dir.mkdir()
        self.rejected()

    def test_outside_output(self):
        self.args.output_dir = self.root / 'wrong'
        self.rejected()

    def test_any_compute_pid_blocks(self):
        self.gpu.return_value['pids'] = [61421]
        self.rejected()

    def test_large_used_memory_blocks_even_without_pid(self):
        self.gpu.return_value['used_mb'] = 23575
        self.rejected()

    def test_utilization_blocks(self):
        self.gpu.return_value['utilization'] = 11
        self.rejected()

    def test_no_cpu_evidence(self):
        self.args.cpu_witness = None
        self.rejected()

    def test_no_cuda_profile(self):
        self.args.profile_dir = None
        self.rejected()

    def test_incomplete_copy(self):
        (self.env.parent / 'copy_status.json').write_text(json.dumps({'source_exit': 1, 'extract_exit': 0}))
        self.rejected()

    def test_launch_binds_only_physical_gpu0_and_detaches(self):
        m.launch(self.args)
        self.assertEqual(self.check.call_count, 2)
        call = self.popen.call_args
        self.assertEqual(call[1]['env']['CUDA_VISIBLE_DEVICES'], m.UUID)
        self.assertTrue(call[1]['start_new_session'])
        self.assertIn('--worker', call[0][0])
        self.assertTrue((self.args.output_dir / 'deployment.json').exists())

    def test_cpu_launch_hides_cuda_even_when_gpu_busy(self):
        self.args.mode = 'cpu'
        self.gpu.return_value['pids'] = [61421]
        m.launch(self.args)
        self.assertEqual(self.popen.call_args[1]['env']['CUDA_VISIBLE_DEVICES'], '')

    def test_shared_exact_pid_allowed_with_headroom(self):
        self.args.share_with_pid = 61421
        self.gpu.return_value['pids'] = [61421]
        m.launch(self.args)
        state = json.loads((self.args.output_dir / 'deployment.json').read_text())
        self.assertEqual(state['placement'], {'shared_with_pid': 61421,
            'allocator_mb': 6144, 'duty_cycle': .20})
        self.assertIn('--share-with-pid', self.popen.call_args[0][0])

    def test_shared_changed_pid_rejected(self):
        self.args.share_with_pid = 61421
        self.gpu.return_value['pids'] = [999]
        self.rejected()

    def test_shared_extra_pid_rejected(self):
        self.args.share_with_pid = 61421
        self.gpu.return_value['pids'] = [61421, 999]
        self.rejected()

    def test_shared_no_external_pid_rejected(self):
        self.args.share_with_pid = 61421
        self.rejected()

    def test_shared_low_headroom_rejected(self):
        self.args.share_with_pid = 61421
        self.gpu.return_value.update(pids=[61421], free_mb=12287)
        self.rejected()

    def test_negative_external_pid_rejected(self):
        self.args.share_with_pid = -1
        self.rejected()


if __name__ == '__main__':
    unittest.main()
