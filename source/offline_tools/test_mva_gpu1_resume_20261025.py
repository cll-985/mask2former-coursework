"""CPU-only migration routing tests; no CUDA initialization or dataset access."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mva_gpu1_resume_20261025 as adapter


class Routing(unittest.TestCase):
    def idle(self):
        return dict(uuid=adapter.UUID, pids=[], used_mb=8, free_mb=24260, utilization=0)

    def test_latest_full_source(self):
        self.assertIn('20261023_gpu0_resume_v2/stage/resource_stop.pth', str(adapter.engine.SOURCE))
        self.assertEqual(adapter.engine.SOURCE_ITERATION, 236751)
        self.assertEqual(adapter.engine.SOURCE_SHA, '3182a3879513ac59e6145271a5fcd846e0b26e5c166b4280ff835e2618e5add5')

    def test_binding_and_inventory_identity(self):
        self.assertEqual(adapter.engine.UUID, adapter.UUID)
        self.assertEqual(adapter.engine.previous.UUID, adapter.UUID)

    def test_detached_child_uses_gpu1_adapter(self):
        self.assertEqual(Path(adapter.engine.__file__).name, 'run_mva_gpu1_resume_20261025.py')

    def test_idle_accepted(self):
        adapter.require_idle(self.idle())

    def test_busy_or_wrong_gpu_rejected(self):
        for change in [dict(pids=[991]), dict(used_mb=500), dict(utilization=11), dict(uuid='wrong')]:
            gpu = self.idle(); gpu.update(change)
            with self.assertRaises(RuntimeError):
                adapter.require_idle(gpu)

    def test_old_and_new_workers_block_duplicates(self):
        for script in ['run_mva_gpu0_resume_v2.py', 'run_mva_gpu0_resume.py',
                       'run_mva_gpu0_exclusive.py', 'run_mva_gpu1_resume_20261025.py']:
            with patch.object(adapter.engine.native, 'output', return_value=' PID COMMAND ARGS\n999 python3 python3 '+script+' --worker --mode train'):
                self.assertTrue(adapter.duplicate_training())

    def test_launcher_is_not_worker(self):
        with patch.object(adapter.engine.native, 'output', return_value=' PID COMMAND ARGS\n999 python3 python3 run_mva_gpu1_resume_20261025.py --mode train --launch'):
            self.assertFalse(adapter.duplicate_training())

    def test_existing_shared_policy_keeps_safety(self):
        self.assertEqual(adapter.engine.sharing_duty(dict(pids=[11],free_mb=20000),11),1.)
        self.assertEqual(adapter.engine.sharing_duty(dict(pids=[11,22],free_mb=16000),11),.20)
        with self.assertRaises(adapter.engine.native.Pressure):
            adapter.engine.sharing_duty(dict(pids=[11],free_mb=4095),11)


if __name__ == '__main__':
    unittest.main()
