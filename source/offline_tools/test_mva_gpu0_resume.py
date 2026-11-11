"""CPU-only resume regression: real native serializers, simulated scaler, tiny SGD kernels."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import sys

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('Explicitly hide CUDA for this regression')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mva_training_state import torch, make_outer, cpu_update, DetectionCheckpointer
from mva_resume_state import file_sha, load_checked, restore_full_state
import run_mva_gpu0_resume as entry


class Resume(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory(prefix='mva_resume_test_')
        self.path = Path(self.tmp.name) / 'source.pth'
        self.source = make_outer()
        cpu_update(self.source); cpu_update(self.source)
        self.source.checkpointer = DetectionCheckpointer(self.source.model, self.tmp.name, trainer=self.source)
        self.source.checkpointer.save('source', iteration=6)
        self.target = make_outer(seed=91, scale=8.)
        self.target.checkpointer = DetectionCheckpointer(self.target.model, trainer=self.target)

    def tearDown(self):
        self.tmp.cleanup()

    def test_restore_and_next_native_cpu_update(self):
        result = restore_full_state(self.target, self.path, torch, file_sha(self.path), 6)
        self.assertEqual(result['start_iter'], 7)
        self.assertEqual(self.target.start_iter, 7)
        self.assertEqual(result['optimizer_resume_status'], 'complete')
        self.assertFalse(result['dataloader_position_restored'])
        cpu_update(self.source); cpu_update(self.target)
        for a,b in zip(self.source.model.parameters(),self.target.model.parameters()):
            self.assertTrue(torch.equal(a,b))

    def test_wrong_sha_rejected_before_mutation(self):
        before = self.target.model.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            restore_full_state(self.target, self.path, torch, 'bad')
        self.assertTrue(torch.equal(before,self.target.model.weight))

    def test_wrong_counter_rejected(self):
        with self.assertRaisesRegex(ValueError, 'iteration'):
            restore_full_state(self.target,self.path,torch,expected_iteration=99)

    def test_legacy_rejected_before_model_mutation(self):
        checkpoint = torch.load(str(self.path),map_location='cpu')
        del checkpoint['trainer']['_trainer']
        torch.save(checkpoint,str(self.path))
        before = self.target.model.weight.detach().clone()
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            restore_full_state(self.target,self.path,torch)
        self.assertTrue(torch.equal(before,self.target.model.weight))

    def test_nonfinite_model_rejected(self):
        checkpoint = torch.load(str(self.path),map_location='cpu')
        checkpoint['model']['weight'].fill_(float('nan'))
        torch.save(checkpoint,str(self.path))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            load_checked(self.path,torch)

    def test_shape_mismatch_rejected(self):
        checkpoint = torch.load(str(self.path),map_location='cpu')
        checkpoint['model']['weight'] = torch.zeros(2,3)
        torch.save(checkpoint,str(self.path))
        with self.assertRaisesRegex(ValueError,'shapes'):
            restore_full_state(self.target,self.path,torch)

    def test_nonfinite_momentum_rejected(self):
        checkpoint = torch.load(str(self.path),map_location='cpu')
        next(iter(checkpoint['trainer']['_trainer']['optimizer']['state'].values()))['momentum_buffer'].fill_(float('inf'))
        torch.save(checkpoint,str(self.path))
        with self.assertRaisesRegex(ValueError,'momentum'):
            load_checked(self.path,torch)

    def test_new_peer_is_not_a_stop(self):
        gpu={'pids':[11,22], 'free_mb':16000}
        self.assertEqual(entry.sharing_duty(gpu,11),.20)
        gpu['pids']=[11,22,33]
        self.assertEqual(entry.sharing_duty(gpu,11),.20)

    def test_peer_disappearance_allows_full_speed(self):
        self.assertEqual(entry.sharing_duty({'pids':[11], 'free_mb':20000},11),1.)

    def test_low_memory_still_stops_ours(self):
        with self.assertRaises(entry.native.Pressure):
            entry.sharing_duty({'pids':[11,22], 'free_mb':4095},11)

    def test_launcher_itself_is_not_duplicate_worker(self):
        with patch.object(entry.native,'output',return_value=' PID COMMAND ARGS\n999 python3 python3 run_mva_gpu0_resume.py --mode train --launch'):
            self.assertFalse(entry.duplicate_training())

    def test_existing_train_worker_rejected(self):
        with patch.object(entry.native,'output',return_value=' PID COMMAND ARGS\n999 python3 python3 run_mva_gpu0_resume.py --worker --mode train'):
            self.assertTrue(entry.duplicate_training())


if __name__ == '__main__':
    unittest.main()
