"""Synthetic safety checks. Never import CUDA libraries or query real GPUs."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('gpu2', Path(__file__).with_name('run_mva_gpu2_native.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Safety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = self.root/'env/conda'; (self.env/'bin').mkdir(parents=True)
        (self.env/'bin/python').write_text('fixture')
        (self.env.parent/'copy_status.json').write_text(json.dumps({'source_exit':0,'extract_exit':0}))
        (self.root/'output_offline').mkdir()
        self.args = argparse.Namespace(mode='cpu',output_dir=self.root/'output_offline/new',
            cpu_witness=None, profile_dir=None, launch=False)
        self.stack=contextlib.ExitStack()
        for name,value in [('ROOT',self.root),('ENV',self.env)]:
            self.stack.enter_context(patch.object(m,name,value))
        self.stack.enter_context(patch.object(m,'verify_sources',return_value={'f':'hash'}))
        self.popen=self.stack.enter_context(patch.object(m.subprocess,'Popen'))
        self.gpu=self.stack.enter_context(patch.object(m,'inventory',return_value=(20000,{m.EXTERNAL_PID})))
        self.evidence=self.stack.enter_context(patch.object(m,'check_evidence'))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    def tearDown(self):
        self.stack.close(); self.temp.cleanup()
    def test_dry_run_no_writes(self):
        m.launch(self.args); self.assertFalse(self.args.output_dir.exists()); self.popen.assert_not_called()
    def test_existing_output_rejected(self):
        self.args.output_dir.mkdir()
        with self.assertRaises(RuntimeError):m.launch(self.args)
    def test_outside_output_rejected(self):
        self.args.output_dir=self.root/'wrong'
        with self.assertRaises(ValueError):m.launch(self.args)
    def test_changed_pid_rejected(self):
        self.args.mode='profile'; self.gpu.return_value=(20000,{999})
        with self.assertRaises(RuntimeError):m.launch(self.args)
    def test_extra_pid_rejected(self):
        self.args.mode='profile'; self.gpu.return_value=(20000,{m.EXTERNAL_PID,999})
        with self.assertRaises(RuntimeError):m.launch(self.args)
    def test_headroom_rejected(self):
        self.args.mode='profile'; self.gpu.return_value=(12000,{m.EXTERNAL_PID})
        with self.assertRaises(RuntimeError):m.launch(self.args)
    def test_profile_needs_cpu_witness(self):
        self.args.mode='profile'
        with self.assertRaises(RuntimeError):m.launch(self.args)
    def test_train_needs_profile(self):
        self.args.mode='train';self.args.cpu_witness=self.root/'witness'
        with self.assertRaises(RuntimeError):m.launch(self.args)
    def test_partial_copy_rejected(self):
        (self.env.parent/'copy_status.json').write_text(json.dumps({'source_exit':1,'extract_exit':0}))
        with self.assertRaises(RuntimeError):m.launch(self.args)
    def test_native_binding(self):
        self.assertEqual(m.native_env('profile',self.root)['CUDA_VISIBLE_DEVICES'],m.UUID)
        self.assertEqual(m.native_env('cpu',self.root)['CUDA_VISIBLE_DEVICES'],'')
    def test_slowdown_needs_two_full_intervals(self):
        self.assertFalse(m.external_slow([{'epoch':10,'time':0},{'epoch':11,'time':2000},
                                          {'epoch':12,'time':4000}],10))
        self.assertTrue(m.external_slow([{'epoch':11,'time':0},{'epoch':12,'time':2000},
                                         {'epoch':13,'time':4000}],10))
    def test_normal_progress_allowed(self):
        self.assertFalse(m.external_slow([{'epoch':11,'time':0},{'epoch':12,'time':1044},
                                          {'epoch':13,'time':2088}],10))
    def test_duty_default_and_bounds(self):
        self.assertEqual(m.policy_duty(.4,[],0),.4)
        for value in [.1,.6,float('nan')]:
            with self.assertRaises(m.Pressure):m.policy_duty(value,[],0)
    def test_duty_reduction(self):
        rows=[{'epoch':11,'time':0},{'epoch':12,'time':1200}]
        self.assertEqual(m.policy_duty(.4,rows,10),.2)
    def test_duty_growth_needs_full_intervals(self):
        rows=[{'epoch':11,'time':0},{'epoch':12,'time':1044},{'epoch':13,'time':2088}]
        self.assertEqual(m.policy_duty(.4,rows,10),.5)
        self.assertEqual(m.policy_duty(.4,rows,11),.4)
    def test_exclusive_requires_verified_completion(self):
        self.assertFalse(m.clean_exclusive({42},[{'epoch':299}],42))
        self.assertFalse(m.clean_exclusive({42,99},[{'epoch':300}],42))
        self.assertTrue(m.clean_exclusive({42},[{'epoch':300}],42))
        self.assertEqual(m.policy_duty(.4,[],0,exclusive=True),1.)


if __name__=='__main__':unittest.main()
