"""Regression tests for UF filtering and negative group fractions."""

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.stats import mannwhitneyu

from spaDIVA.interpretation import (
    add_unique_factor_ratio,
    summarize_extreme_groups,
)


class UniqueFactorBoundaryTests(unittest.TestCase):
    def test_nonpositive_and_below_threshold_joint_r2_remain_undefined(self):
        frame = pd.DataFrame({
            "r2_z": [-0.3, -0.1, 0.0, 0.0],
            "r2_zw": [-0.1, 0.0, 0.0005, 0.001],
        })
        result = add_unique_factor_ratio(frame)
        self.assertTrue(result["unique_factor_ratio"].isna().all())
        self.assertEqual(len(result), len(frame))

    def test_negative_gain_is_zero_and_negative_shared_r2_is_retained(self):
        frame = pd.DataFrame({"r2_z": [0.5, -0.2], "r2_zw": [0.4, 0.4]})
        result = add_unique_factor_ratio(frame)
        assert_allclose(result["unique_factor_gain"], [0.0, 0.6])
        assert_allclose(result["unique_factor_ratio"], [0.0, 1.5])

    def test_negative_fraction_is_rejected(self):
        frame = pd.DataFrame({
            "unique_factor_ratio": np.linspace(0.0, 1.0, 10),
            "external_feature_pearson": np.linspace(1.0, 0.0, 10),
        })
        for fraction in (-1.0, -0.2, -1e-12):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "fraction must be non-negative"):
                    summarize_extreme_groups(frame, fraction=fraction)

    def test_default_groups_and_one_sided_test_are_unchanged(self):
        frame = pd.DataFrame({
            "unique_factor_ratio": np.linspace(0.0, 1.0, 10),
            "external_feature_pearson": np.linspace(1.0, 0.0, 10),
        })
        low = frame["external_feature_pearson"].iloc[:2]
        high = frame["external_feature_pearson"].iloc[-2:]
        expected = mannwhitneyu(low, high, alternative="greater")
        result = summarize_extreme_groups(frame)
        self.assertEqual((result["n_low"], result["n_high"]), (2, 2))
        self.assertEqual(result["mannwhitney_alternative"], "greater")
        self.assertEqual(result["mannwhitney_u"], expected.statistic)
        self.assertEqual(result["mannwhitney_p"], expected.pvalue)


if __name__ == "__main__":
    unittest.main()
