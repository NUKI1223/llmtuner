"""Known-answer tests for llmtuner.stats (standard library only)."""

import math
import unittest

from llmtuner.stats import (
    Interval,
    agreement,
    bootstrap_mean,
    mcnemar_p,
    paired_compare,
    paired_detection_floor,
    pass_rate,
    wilson_interval,
)


class PassRateTests(unittest.TestCase):
    def test_42_of_60(self):
        iv = pass_rate([True] * 42 + [False] * 18)
        self.assertAlmostEqual(iv.point, 0.700, places=3)
        self.assertAlmostEqual(iv.low, 0.575, delta=0.003)
        self.assertAlmostEqual(iv.high, 0.801, delta=0.003)
        self.assertTrue(iv.contains(0.7))

    def test_wilson_matches_pass_rate(self):
        self.assertEqual(wilson_interval(42, 60), pass_rate([True] * 42 + [False] * 18))

    def test_bootstrap_mean_42_of_60(self):
        iv = bootstrap_mean([1.0] * 42 + [0.0] * 18)
        self.assertAlmostEqual(iv.point, 0.700, places=3)
        self.assertAlmostEqual(iv.low, 0.583, delta=0.02)
        self.assertAlmostEqual(iv.high, 0.817, delta=0.02)

    def test_bootstrap_deterministic_with_same_seed(self):
        data = [1.0] * 17 + [0.0] * 9
        self.assertEqual(bootstrap_mean(data, seed=7), bootstrap_mean(data, seed=7))

    def test_pass_rate_is_deterministic(self):
        data = [True, False, True, True, False, True, True, True]
        self.assertEqual(pass_rate(data), pass_rate(data))

    def test_empty_is_nan(self):
        for iv in (pass_rate([]), wilson_interval(0, 0)):
            self.assertTrue(math.isnan(iv.point))
            self.assertTrue(math.isnan(iv.low))
            self.assertTrue(math.isnan(iv.high))

    def test_extremes_are_not_degenerate(self):
        one = pass_rate([True])
        self.assertEqual(one.point, 1.0)
        self.assertEqual(one.high, 1.0)
        self.assertLess(one.low, 0.5)
        up = pass_rate([True] * 10)
        self.assertEqual((up.point, up.high), (1.0, 1.0))
        self.assertLess(up.low, 1.0)
        down = pass_rate([False] * 10)
        self.assertEqual((down.point, down.low), (0.0, 0.0))
        self.assertGreater(down.high, 0.0)

    def test_bootstrap_mean_bounds_ordered(self):
        iv = bootstrap_mean([0.0, 1.0, 1.0, 0.0, 1.0])
        self.assertLessEqual(iv.low, iv.point)
        self.assertLessEqual(iv.point, iv.high)

    def test_interval_as_dict(self):
        self.assertEqual(Interval(0.5, 0.4, 0.6).as_dict(), {"point": 0.5, "low": 0.4, "high": 0.6})


class AgreementTests(unittest.TestCase):
    def test_known_answer(self):
        human = [True] * 26 + [True] * 8 + [False] * 16 + [False] * 10
        grader = [True] * 26 + [False] * 8 + [True] * 16 + [False] * 10
        a = agreement(human, grader)
        self.assertEqual(a.n, 60)
        self.assertEqual(a.both_pass, 26)
        self.assertEqual(a.human_pass_grader_fail, 8)
        self.assertEqual(a.human_fail_grader_pass, 16)
        self.assertEqual(a.both_fail, 10)
        self.assertAlmostEqual(a.agreement, 0.60, places=6)
        self.assertAlmostEqual(a.kappa, 0.155, delta=0.005)
        self.assertAlmostEqual(a.catch_rate, 10 / 26)
        self.assertAlmostEqual(a.false_fail_rate, 8 / 34)

    def test_perfect_agreement(self):
        h = [True, False, True, False]
        a = agreement(h, list(h))
        self.assertEqual(a.agreement, 1.0)
        self.assertAlmostEqual(a.kappa, 1.0)
        self.assertEqual(a.catch_rate, 1.0)
        self.assertEqual(a.false_fail_rate, 0.0)

    def test_all_pass_kappa_is_none(self):
        a = agreement([True] * 5, [True] * 5)
        self.assertIsNone(a.kappa)
        self.assertEqual(a.agreement, 1.0)
        self.assertIsNone(a.catch_rate)  # no human-failed outputs
        self.assertEqual(a.false_fail_rate, 0.0)

    def test_all_fail_kappa_is_none(self):
        a = agreement([False] * 5, [False] * 5)
        self.assertIsNone(a.kappa)
        self.assertIsNone(a.false_fail_rate)  # no human-passed outputs

    def test_empty(self):
        a = agreement([], [])
        self.assertEqual(a.n, 0)
        self.assertTrue(math.isnan(a.agreement))
        self.assertIsNone(a.kappa)
        self.assertIsNone(a.catch_rate)
        self.assertIsNone(a.false_fail_rate)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            agreement([True], [True, False])

    def test_as_dict(self):
        self.assertEqual(agreement([True], [True]).as_dict()["n"], 1)


class PairedFloorTests(unittest.TestCase):
    def test_none_below_six_cases(self):
        for n in range(6):
            self.assertIsNone(paired_detection_floor(n))

    def test_six_over_n(self):
        self.assertEqual(paired_detection_floor(6), 1.0)
        self.assertAlmostEqual(paired_detection_floor(60), 0.1)

    def test_six_one_way_is_the_smallest_detectable(self):
        self.assertLess(mcnemar_p(0, 6), 0.05)
        self.assertGreaterEqual(mcnemar_p(0, 5), 0.05)


class McNemarTests(unittest.TestCase):
    def test_no_disagreements(self):
        self.assertEqual(mcnemar_p(0, 0), 1.0)

    def test_balanced_is_one(self):
        self.assertEqual(mcnemar_p(5, 5), 1.0)

    def test_known_exact_value(self):
        # 0 vs 10: two-sided exact p = 2 * (1 / 1024)
        self.assertAlmostEqual(mcnemar_p(0, 10), 2 / 1024)
        self.assertAlmostEqual(mcnemar_p(10, 0), 2 / 1024)

    def test_symmetric(self):
        self.assertEqual(mcnemar_p(3, 9), mcnemar_p(9, 3))

    def test_capped_at_one(self):
        self.assertLessEqual(mcnemar_p(1, 2), 1.0)


class PairedCompareTests(unittest.TestCase):
    def test_run_against_itself(self):
        run = {f"c{i}": (i % 3 != 0) for i in range(30)}
        p = paired_compare(run, dict(run))
        self.assertEqual(p.n, 30)
        self.assertEqual(p.diff.point, 0.0)
        self.assertTrue(p.diff.contains(0.0))
        self.assertEqual(p.regressions, [])
        self.assertEqual(p.improvements, [])
        self.assertFalse(p.measurable)
        self.assertEqual(p.p_value, 1.0)
        self.assertEqual(p.baseline_rate, p.candidate_rate)

    def test_clear_regression(self):
        baseline = {f"c{i:02d}": True for i in range(20)}
        candidate = dict(baseline)
        broken = [f"c{i:02d}" for i in range(15)]
        for i in broken:
            candidate[i] = False
        p = paired_compare(baseline, candidate)
        self.assertTrue(p.measurable)
        self.assertLess(p.diff.point, 0)
        self.assertAlmostEqual(p.diff.point, -0.75)
        self.assertEqual(p.regressions, broken)
        self.assertEqual(p.improvements, [])
        self.assertLess(p.p_value, 0.05)

    def test_improvements_listed(self):
        baseline = {f"c{i:02d}": False for i in range(20)}
        candidate = {f"c{i:02d}": (i < 16) for i in range(20)}
        p = paired_compare(baseline, candidate)
        self.assertTrue(p.measurable)
        self.assertGreater(p.diff.point, 0)
        self.assertEqual(len(p.improvements), 16)
        self.assertEqual(p.regressions, [])

    def test_only_shared_cases_compared(self):
        p = paired_compare({"a": True, "b": True, "x": False}, {"a": True, "b": False, "y": True})
        self.assertEqual(p.n, 2)
        self.assertEqual(p.regressions, ["b"])
        self.assertEqual(p.improvements, [])

    def test_no_overlap(self):
        p = paired_compare({"a": True}, {"b": True})
        self.assertEqual(p.n, 0)
        self.assertFalse(p.measurable)
        self.assertTrue(math.isnan(p.baseline_rate))
        self.assertEqual(p.p_value, 1.0)

    def test_empty(self):
        p = paired_compare({}, {})
        self.assertEqual(p.n, 0)
        self.assertFalse(p.measurable)

    def test_one_shared_regression_is_not_measurable(self):
        p = paired_compare({"a": True}, {"a": False})
        self.assertEqual(p.p_value, 1.0)
        self.assertFalse(p.measurable)

    def test_four_regressions_in_fifty_not_measurable(self):
        baseline = {f"c{i}": True for i in range(50)}
        candidate = {k: (k not in ("c0", "c1", "c2", "c3")) for k in baseline}
        p = paired_compare(baseline, candidate)
        self.assertAlmostEqual(p.p_value, 0.125)
        self.assertFalse(p.measurable)

    def test_six_regressions_are_measurable(self):
        baseline = {f"c{i}": True for i in range(50)}
        candidate = {k: (k not in ("c0", "c1", "c2", "c3", "c4", "c5")) for k in baseline}
        self.assertTrue(paired_compare(baseline, candidate).measurable)

    def test_deterministic(self):
        baseline = {f"c{i}": i % 2 == 0 for i in range(25)}
        candidate = {f"c{i}": i % 3 == 0 for i in range(25)}
        self.assertEqual(paired_compare(baseline, candidate, seed=3), paired_compare(baseline, candidate, seed=3))

    def test_as_dict_includes_measurable(self):
        d = paired_compare({"a": True}, {"a": True}).as_dict()
        self.assertIn("measurable", d)
        self.assertFalse(d["measurable"])


if __name__ == "__main__":
    unittest.main()
