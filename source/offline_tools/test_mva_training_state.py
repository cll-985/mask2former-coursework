"""CPU-only native Detectron2 checkpoint regression; simulated scaler payload."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('This regression must explicitly hide CUDA devices')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.runtime/mva_tv0100cu111'))
sys.path.insert(0, str(ROOT / 'offline_tools'))
import torch
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultTrainer
from detectron2.engine.train_loop import AMPTrainer, TrainerBase
from detectron2.engine.hooks import LRScheduler
from detectron2.checkpoint import DetectionCheckpointer
from mva_training_state import InnerTrainerCheckpointMixin, checkpoint_summary


class ScalerPayload:
    """State transport fixture; NOT a CUDA AMP computation test."""
    def __init__(self, scale):
        self.value = {'scale': scale, 'growth_factor': 2., 'backoff_factor': .5,
                      'growth_interval': 2000, '_growth_tracker': 7}
    def state_dict(self):
        return copy.deepcopy(self.value)
    def load_state_dict(self, state):
        self.value = copy.deepcopy(state)


class Outer(InnerTrainerCheckpointMixin, DefaultTrainer):
    pass


def make_outer(seed=3407, scale=1024.):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    outer = Outer.__new__(Outer)
    TrainerBase.__init__(outer)
    outer.cfg = get_cfg(); outer.cfg.SOLVER.AMP.ENABLED = True
    outer._trainer = AMPTrainer(model, [], opt, grad_scaler=ScalerPayload(scale))
    outer.model, outer.optimizer = model, opt
    outer.scheduler = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=.9)
    outer.register_hooks([LRScheduler(opt, outer.scheduler)])
    outer.iter, outer._trainer.iter = 6, 6
    return outer


def cpu_update(trainer):
    trainer.optimizer.zero_grad()
    x = torch.tensor([[1., 2., 3.], [2., 1., -1.]])
    loss = trainer.model(x).square().mean()
    loss.backward(); trainer.optimizer.step(); trainer.scheduler.step()
    return float(loss.detach())


class Regression(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.a = make_outer()
        self.losses = [cpu_update(self.a), cpu_update(self.a)]
    def test_explicit_inner_state_is_saved(self):
        state = self.a.state_dict()
        self.assertTrue(state['_trainer']['optimizer']['state'])
        self.assertEqual(state['_trainer']['grad_scaler']['scale'], 1024.)
        self.assertTrue(state['hooks']['LRScheduler'])
    def test_legacy_state_rejected_before_counter_change(self):
        state = self.a.state_dict(); del state['_trainer']
        target = make_outer(); target.iter = 99
        with self.assertRaisesRegex(ValueError, 'lacks inner trainer'):
            target.load_state_dict(state)
        self.assertEqual(target.iter, 99)
    def test_missing_optimizer_rejected(self):
        state = self.a.state_dict(); del state['_trainer']['optimizer']
        with self.assertRaisesRegex(ValueError, 'optimizer'):
            make_outer().load_state_dict(state)
    def test_missing_scaler_rejected(self):
        state = self.a.state_dict(); del state['_trainer']['grad_scaler']
        with self.assertRaisesRegex(ValueError, 'GradScaler'):
            make_outer().load_state_dict(state)
    def test_actual_native_checkpoint_and_next_cpu_update(self):
        with tempfile.TemporaryDirectory(prefix='mva_cpu_state_') as d:
            DetectionCheckpointer(self.a.model, d, trainer=self.a).save('roundtrip', iteration=6)
            p = Path(d) / 'roundtrip.pth'
            summary = checkpoint_summary(torch.load(str(p), map_location='cpu'))
            self.assertEqual(summary['optimizer_resume_status'], 'complete')
            target = make_outer(seed=91, scale=8.)
            DetectionCheckpointer(target.model, trainer=target).load(str(p))
            self.assertEqual(target.iter, 6)
            self.assertEqual(target._trainer.grad_scaler.state_dict(), self.a._trainer.grad_scaler.state_dict())
            self.assertEqual(target.scheduler.state_dict(), self.a.scheduler.state_dict())
            self.assertEqual(target.optimizer.param_groups[0]['lr'], self.a.optimizer.param_groups[0]['lr'])
            for p0,p1 in zip(self.a.model.parameters(), target.model.parameters()):
                self.assertTrue(torch.equal(p0, p1))
                self.assertTrue(torch.equal(self.a.optimizer.state[p0]['momentum_buffer'],
                                            target.optimizer.state[p1]['momentum_buffer']))
            cpu_update(self.a); cpu_update(target)
            for p0,p1 in zip(self.a.model.parameters(), target.model.parameters()):
                self.assertTrue(torch.equal(p0,p1))
    def test_legacy_summary_is_incomplete(self):
        state = self.a.state_dict(); del state['_trainer']
        summary = checkpoint_summary({'model': self.a.model.state_dict(), 'trainer':state, 'iteration':6})
        self.assertEqual(summary['optimizer_resume_status'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
