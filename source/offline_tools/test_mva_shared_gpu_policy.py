import unittest
from mva_shared_gpu_policy import SharingPolicy


class PolicyTests(unittest.TestCase):
    def test_start_includes_context_and_reserve(self):
        p = SharingPolicy()
        self.assertEqual(p.start_free_mb, 7680)
        self.assertFalse(p.start_allowed(7679))
        self.assertTrue(p.start_allowed(7680))

    def test_step_reserve(self):
        self.assertFalse(SharingPolicy().step_allowed(2047))
        self.assertTrue(SharingPolicy().step_allowed(2048))

    def test_fifth_duty_budget(self):
        self.assertAlmostEqual(SharingPolicy().delay(1), 4)

    def test_no_excessive_memory_budget(self):
        with self.assertRaises(ValueError):
            SharingPolicy(memory_limit_mb=4096)

    def test_no_excessive_duty(self):
        with self.assertRaises(ValueError):
            SharingPolicy(duty_cycle=.5)

    def test_no_small_reserve(self):
        with self.assertRaises(ValueError):
            SharingPolicy(reserve_mb=1024)

    def test_old_context_allowance_is_unsafe(self):
        with self.assertRaises(ValueError):
            SharingPolicy(context_allowance_mb=512)

    def test_initial_probe_free_memory_would_now_be_rejected(self):
        self.assertFalse(SharingPolicy().start_allowed(7555))


if __name__ == '__main__':
    unittest.main()
