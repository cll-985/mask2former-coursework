"""Latest checkpoint selection and child/duplicate routing:CPU-only synthetic tests."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mva_gpu0_resume_v2 as adapter


class Routing(unittest.TestCase):
    def test_latest_checkpoint_selected(self):
        self.assertIn('20261023_gpu0_resume_v1/stage/resource_stop.pth', str(adapter.engine.SOURCE))
        self.assertEqual(adapter.engine.SOURCE_ITERATION,109275)
        self.assertEqual(adapter.engine.SOURCE_SHA,'ec5968dbd82252ea220dd62b900acc8cc25f24babbdedf05ca895ddda4fd5202')

    def test_child_uses_adapter(self):
        self.assertEqual(Path(adapter.engine.__file__).name,'run_mva_gpu0_resume_v2.py')

    def test_v2_duplicate_blocked(self):
        with patch.object(adapter.engine.native,'output',return_value=' PID COMMAND ARGS\n999 python3 python3 run_mva_gpu0_resume_v2.py --worker --mode train'):
            self.assertTrue(adapter.duplicate_training())

    def test_historical_duplicate_blocked(self):
        with patch.object(adapter.engine.native,'output',return_value=' PID COMMAND ARGS\n999 python3 python3 run_mva_gpu0_resume.py --worker --mode train'):
            self.assertTrue(adapter.duplicate_training())

    def test_launcher_not_duplicate(self):
        with patch.object(adapter.engine.native,'output',return_value=' PID COMMAND ARGS\n999 python3 python3 run_mva_gpu0_resume_v2.py --mode train --launch'):
            self.assertFalse(adapter.duplicate_training())


if __name__=='__main__':
    unittest.main()
