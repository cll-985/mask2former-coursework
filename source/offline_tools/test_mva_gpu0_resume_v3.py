"""CPU-only adapter routing and unchanged resource-policy checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mva_gpu0_resume_v3 as adapter


class Routing(unittest.TestCase):
    def test_latest_source_pin(self):
        self.assertIn('20261025_gpu1_resume_v1/stage/resource_stop.pth', str(adapter.engine.SOURCE))
        self.assertEqual(adapter.engine.SOURCE_ITERATION,321627)
        self.assertEqual(adapter.engine.SOURCE_SHA,'587c7776b2c6560f5eb9da9ab6990e1618d6782ad0be675e471adba878a7fa4e')

    def test_gpu_binding_and_inventory(self):
        self.assertEqual(adapter.engine.UUID,adapter.UUID)
        self.assertEqual(adapter.engine.previous.UUID,adapter.UUID)
        self.assertEqual(adapter.UUID,'GPU-a332beaa-9bf1-c84b-f27c-8bdcaaebed4c')

    def test_child_uses_v3(self):
        self.assertEqual(Path(adapter.engine.__file__).name,'run_mva_gpu0_resume_v3.py')

    def test_all_historical_workers_blocked(self):
        for script in ['run_mva_gpu0_resume.py','run_mva_gpu0_resume_v2.py',
                       'run_mva_gpu0_resume_v3.py','run_mva_gpu0_exclusive.py',
                       'run_mva_gpu1_resume_20261025.py']:
            with patch.object(adapter.engine.native,'output',return_value=' PID COMMAND ARGS\n999 python3 python3 '+script+' --worker --mode train'):
                self.assertTrue(adapter.duplicate_training())

    def test_launcher_not_worker(self):
        with patch.object(adapter.engine.native,'output',return_value=' PID COMMAND ARGS\n999 python3 python3 run_mva_gpu0_resume_v3.py --mode train --launch'):
            self.assertFalse(adapter.duplicate_training())

    def test_same_cooperative_duty(self):
        self.assertEqual(adapter.engine.sharing_duty({'pids':[11,22],'free_mb':16000},11),.20)
        self.assertEqual(adapter.engine.sharing_duty({'pids':[11],'free_mb':16000},11),1.)

    def test_reserve_not_relaxed(self):
        with self.assertRaises(adapter.engine.native.Pressure):
            adapter.engine.sharing_duty({'pids':[11],'free_mb':4095},11)


if __name__ == '__main__':
    unittest.main()
