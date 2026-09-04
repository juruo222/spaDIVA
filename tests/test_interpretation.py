"""Regression tests for aligned translation inputs and per-feature statistics.

Run from the repository root with:
    python -m unittest discover -s tests -p test_interpretation.py
"""

from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from spaDIVA.interpretation import (
    evaluate_unique_factor_ratio,
    featurewise_pearson,
    featurewise_r2,
    prepare_feature_translation_inputs,
    ridge_predict,
    select_shared_latent,
)


def evaluate_prepared(prepared, **kwargs):
    fields = (
        "Z_train", "W_train", "Y_train", "Z_test", "W_test", "Y_test_true",
        "external_pred", "feature_names", "feature_indices", "z_source",
        "model_seed", "poe_sample_seed",
    )
    return evaluate_unique_factor_ratio(
        **{name: prepared[name] for name in fields}, **kwargs
    )


class SharedLatentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.values = {
            key: np.arange(16, dtype=np.float32).reshape(8, 2) + offset
            for key, offset in (("Z_poe", 0), ("Z_PoE", 100), ("Z", 200), ("Z_WNN", 300))
        }

    def latent_with(self, keys):
        adata = sc.AnnData(np.zeros((8, 1), dtype=np.float32))
        for key in keys:
            adata.obsm[key] = self.values[key].copy()
        return adata

    def test_default_prefers_poe_before_any_wnn_key(self):
        cases = (
            (("Z_poe",), "Z_poe"),
            (("Z_PoE",), "Z_PoE"),
            (("Z_PoE", "Z"), "Z_PoE"),
            (("Z_PoE", "Z_WNN"), "Z_PoE"),
            (("Z_poe", "Z_WNN"), "Z_poe"),
            (("Z_PoE", "Z_WNN", "Z_poe", "Z"), "Z_poe"),
        )
        for keys, expected in cases:
            for use in (None, "Z_poe", "Z_PoE", " z_PoE "):
                with self.subTest(keys=keys, use=use):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        matrix, source = select_shared_latent(self.latent_with(keys), use=use)
                    self.assertEqual(source, expected)
                    assert_array_equal(matrix, self.values[expected])
                    self.assertEqual(len(caught), 0)

    def test_explicit_wnn_uses_standard_key_before_alias_and_never_poe(self):
        cases = (
            (("Z",), "Z"),
            (("Z_WNN",), "Z_WNN"),
            (("Z_poe", "Z_WNN"), "Z_WNN"),
            (("Z_WNN", "Z_PoE", "Z"), "Z"),
        )
        for keys, expected in cases:
            for use in ("Z", " z ", "Z_WNN", " z_wnn "):
                with self.subTest(keys=keys, use=use):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        matrix, source = select_shared_latent(self.latent_with(keys), use=use)
                    self.assertEqual(source, expected)
                    assert_array_equal(matrix, self.values[expected])
                    self.assertEqual(len(caught), 0)

    def test_poe_fallback_warns_and_returns_actual_wnn_key(self):
        for keys, expected in ((("Z",), "Z"), (("Z_WNN",), "Z_WNN"), (("Z_WNN", "Z"), "Z")):
            for use in (None, "Z_poe", "Z_PoE"):
                with self.subTest(keys=keys, use=use):
                    with self.assertWarnsRegex(RuntimeWarning, expected):
                        matrix, source = select_shared_latent(self.latent_with(keys), use=use)
                    self.assertEqual(source, expected)
                    assert_array_equal(matrix, self.values[expected])

    def test_explicit_wnn_does_not_fall_back_to_poe(self):
        for keys in ((), ("Z_poe",), ("Z_PoE",), ("Z_poe", "Z_PoE")):
            for use in ("Z", "Z_WNN"):
                with self.subTest(keys=keys, use=use):
                    with self.assertRaisesRegex(KeyError, "Z_WNN"):
                        select_shared_latent(self.latent_with(keys), use=use)

    def test_missing_keys_and_invalid_selectors_have_clear_errors(self):
        with self.assertRaisesRegex(KeyError, "shared latent"):
            select_shared_latent(self.latent_with(()))
        for use in ("", "poe", "Z_poe_mean", "unknown"):
            with self.subTest(use=use):
                with self.assertRaises(ValueError):
                    select_shared_latent(self.latent_with(self.values), use=use)
        for use in (42, False, [], {}):
            with self.subTest(use=use):
                with self.assertRaises(TypeError):
                    select_shared_latent(self.latent_with(self.values), use=use)

    def test_alias_selection_does_not_add_keys_or_modify_arrays(self):
        adata = self.latent_with(("Z_PoE", "Z_WNN"))
        before = {key: value.copy() for key, value in adata.obsm.items()}
        for use in (None, "Z", "Z_PoE", "Z_WNN"):
            select_shared_latent(adata, use=use)
        self.assertEqual(set(adata.obsm), set(before))
        for key in before:
            assert_array_equal(adata.obsm[key], before[key])

    def test_legacy_h5ad_keys_are_read_without_rewriting_file(self):
        adata = self.latent_with(("Z_PoE", "Z_WNN"))
        with tempfile.TemporaryDirectory(prefix="spadiva_legacy_latent_") as temporary:
            path = Path(temporary) / "latent.h5ad"
            adata.write_h5ad(path)
            before = path.read_bytes()
            for use, expected in ((None, "Z_PoE"), ("Z", "Z_WNN")):
                matrix, source = select_shared_latent(path, use=use)
                self.assertEqual(source, expected)
                assert_array_equal(matrix, self.values[expected])
            self.assertEqual(path.read_bytes(), before)


class TranslationInputTests(unittest.TestCase):
    def setUp(self):
        self.names = pd.Index(list("ABCDEFGH"))
        t = np.arange(1, 9, dtype=np.float32)
        values = np.column_stack((2 * t + np.sin(t), t ** 2 + 1, 3 * t + 4))
        self.features = sc.AnnData(
            values,
            obs=pd.DataFrame(index=self.names),
            var=pd.DataFrame(index=["f0", "f1", "f2"]),
        )
        self.latent = sc.AnnData(
            np.zeros((len(t), 1), dtype=np.float32),
            obs=pd.DataFrame(index=self.names),
        )
        self.latent.obsm["Z_poe"] = np.column_stack((t, t ** 2 / 10))
        self.latent.obsm["Z"] = np.column_stack((100 + t, 200 + t))
        self.latent.obsm["W_ADT"] = np.column_stack((t / 2, np.sin(t)))
        self.latent.uns["model_seed"] = 42
        self.latent.uns["poe_sample_seed"] = 17
        # External predictions intentionally contain held-out observations only.
        self.truth = self.features[list("HFEG"), ["f2", "f0", "f1"]].copy()
        self.prediction = self.features[list("GEHF"), ["f1", "f2", "f0"]].copy()
        self.index_split = {"idx_train": np.arange(4), "idx_test": np.arange(4, 8)}
        self.named_split = {"train_names": list("ABCD"), "test_names": list("EFGH")}

    def prepare(self, split=None, **overrides):
        args = {
            "latent_adata": self.latent,
            "feature_adata": self.features,
            "external_true_adata": self.truth,
            "external_pred_adata": self.prediction,
            "split": self.index_split if split is None else split,
        }
        args.update(overrides)
        return prepare_feature_translation_inputs(**args)

    def assert_expected_alignment(self, result, train="ABCD", test="EFGH"):
        train, test = list(train), list(test)
        self.assertEqual(list(result["train_names"]), train)
        self.assertEqual(list(result["test_names"]), test)
        self.assertEqual(result["z_source"], "Z_poe")
        self.assertEqual(result["model_seed"], 42)
        self.assertEqual(result["poe_sample_seed"], 17)
        assert_array_equal(result["Z_train"], self.latent[train].obsm["Z_poe"])
        assert_array_equal(result["Z_test"], self.latent[test].obsm["Z_poe"])
        assert_array_equal(result["W_train"], self.latent[train].obsm["W_ADT"])
        assert_array_equal(result["Y_train"], self.features[train].X)
        assert_array_equal(result["Y_test_true"], self.features[test].X)
        assert_array_equal(result["external_pred"], self.features[test].X)
        self.assertEqual(list(result["feature_names"]), ["f0", "f1", "f2"])

    def test_integer_split_uses_original_feature_order_before_latent_alignment(self):
        reordered = self.latent[list("EFABCDGH")].copy()
        result = self.prepare(latent_adata=reordered)
        self.assert_expected_alignment(result)
        self.assertFalse(set(result["train_names"]) & set(result["test_names"]))
        # Preparing inputs must not reorder either caller-owned AnnData object.
        self.assertEqual(list(reordered.obs_names), list("EFABCDGH"))
        self.assertEqual(list(self.features.obs_names), list("ABCDEFGH"))

    def test_named_split_is_independent_of_latent_and_feature_order(self):
        result = self.prepare(
            split=self.named_split,
            latent_adata=self.latent[list("HGFEDCBA")].copy(),
            feature_adata=self.features[list("DCBAHGFE")].copy(),
        )
        self.assert_expected_alignment(result)

    def test_integer_split_respects_nonstandard_original_feature_order(self):
        # These positions refer to feature_adata's supplied order, not sorted names.
        reordered_features = self.features[list("DABCHEFG")].copy()
        result = self.prepare(feature_adata=reordered_features)
        self.assert_expected_alignment(result, train="DABC", test="HEFG")

    def test_names_take_priority_over_inconsistent_or_invalid_integer_indices(self):
        split = dict(self.named_split, idx_train=np.array([-1]), idx_test=np.array([100]))
        self.assert_expected_alignment(self.prepare(split=split))

    def test_explicit_Z_selects_wnn_even_when_Z_poe_exists(self):
        prepared = self.prepare(use="Z")
        self.assertEqual(prepared["z_source"], "Z")
        self.assertEqual(prepared["model_seed"], 42)
        self.assertIsNone(prepared["poe_sample_seed"])
        assert_array_equal(prepared["Z_train"], self.latent[list("ABCD")].obsm["Z"])
        assert_array_equal(prepared["Z_test"], self.latent[list("EFGH")].obsm["Z"])
        result = evaluate_prepared(prepared)
        self.assertTrue(result["z_source"].eq("Z").all())
        self.assertNotIn("poe_sample_seed", result.columns)

    def test_default_falls_back_to_Z_without_reporting_poe_sample_seed(self):
        latent = self.latent.copy()
        del latent.obsm["Z_poe"]
        with self.assertWarnsRegex(RuntimeWarning, "Z_poe"):
            prepared = self.prepare(latent_adata=latent)
        self.assertEqual(prepared["z_source"], "Z")
        self.assertEqual(prepared["model_seed"], 42)
        self.assertIsNone(prepared["poe_sample_seed"])
        assert_array_equal(prepared["Z_train"], self.latent[list("ABCD")].obsm["Z"])
        assert_array_equal(prepared["Z_test"], self.latent[list("EFGH")].obsm["Z"])
        result = evaluate_prepared(prepared)
        self.assertTrue(result["z_source"].eq("Z").all())
        self.assertNotIn("poe_sample_seed", result.columns)

    def test_legacy_poe_preserves_statistics_actual_source_and_sample_seed(self):
        expected = evaluate_prepared(self.prepare())
        latent = self.latent.copy()
        latent.obsm["Z_PoE"] = latent.obsm.pop("Z_poe")
        latent.obsm["Z_WNN"] = latent.obsm.pop("Z")
        prepared = self.prepare(latent_adata=latent)
        matrix, source = select_shared_latent(latent)
        self.assertEqual(prepared["z_source"], source)
        self.assertEqual(source, "Z_PoE")
        assert_array_equal(prepared["Z_train"], matrix[:4])
        assert_array_equal(prepared["Z_test"], matrix[4:])
        self.assertEqual(prepared["model_seed"], 42)
        self.assertEqual(prepared["poe_sample_seed"], 17)
        result = evaluate_prepared(prepared)
        self.assertTrue(result["z_source"].eq("Z_PoE").all())
        self.assertTrue(result["poe_sample_seed"].eq(17).all())
        pd.testing.assert_frame_equal(result.drop(columns="z_source"), expected.drop(columns="z_source"))

    def test_legacy_poe_takes_priority_over_standard_wnn_in_preparation(self):
        latent = self.latent.copy()
        latent.obsm["Z_PoE"] = latent.obsm.pop("Z_poe")
        prepared = self.prepare(latent_adata=latent)
        self.assertEqual(prepared["z_source"], "Z_PoE")
        self.assertEqual(prepared["poe_sample_seed"], 17)
        assert_array_equal(prepared["Z_train"], self.latent.obsm["Z_poe"][:4])

    def test_explicit_legacy_wnn_selection_does_not_report_poe_seed(self):
        latent = self.latent.copy()
        latent.obsm["Z_WNN"] = latent.obsm.pop("Z")
        for use in ("Z", "Z_WNN"):
            with self.subTest(use=use):
                prepared = self.prepare(latent_adata=latent, use=use)
                self.assertEqual(prepared["z_source"], "Z_WNN")
                self.assertIsNone(prepared["poe_sample_seed"])
                assert_array_equal(prepared["Z_train"], self.latent.obsm["Z"][:4])
                result = evaluate_prepared(prepared)
                self.assertTrue(result["z_source"].eq("Z_WNN").all())
                self.assertNotIn("poe_sample_seed", result.columns)

    def test_default_legacy_wnn_fallback_preserves_actual_source(self):
        latent = self.latent.copy()
        latent.obsm["Z_WNN"] = latent.obsm.pop("Z")
        del latent.obsm["Z_poe"]
        with self.assertWarnsRegex(RuntimeWarning, "Z_WNN"):
            prepared = self.prepare(latent_adata=latent)
        self.assertEqual(prepared["z_source"], "Z_WNN")
        self.assertIsNone(prepared["poe_sample_seed"])
        result = evaluate_prepared(prepared)
        self.assertTrue(result["z_source"].eq("Z_WNN").all())
        self.assertNotIn("poe_sample_seed", result.columns)

    def test_legacy_poe_without_sample_seed_does_not_invent_one(self):
        latent = self.latent.copy()
        latent.obsm["Z_PoE"] = latent.obsm.pop("Z_poe")
        del latent.uns["poe_sample_seed"]
        prepared = self.prepare(latent_adata=latent)
        self.assertEqual(prepared["z_source"], "Z_PoE")
        self.assertIsNone(prepared["poe_sample_seed"])
        self.assertNotIn("poe_sample_seed", evaluate_prepared(prepared).columns)

    def test_npz_path_accepts_names_and_preserves_names_priority(self):
        with tempfile.TemporaryDirectory(prefix="spadiva_split_") as temporary:
            path = Path(temporary) / "split.npz"
            np.savez(
                path,
                train_names=np.array(list("ABCD"), dtype=object),
                test_names=np.array(list("EFGH"), dtype=object),
                idx_train=np.array([-1]),
                idx_test=np.array([100]),
            )
            result = self.prepare(split=path, latent_adata=self.latent[::-1].copy())
            self.assert_expected_alignment(result)
            # On Windows this also catches an unclosed NPZ file handle.
            path.unlink()

    def test_npz_path_with_integer_split_uses_feature_order(self):
        with tempfile.TemporaryDirectory(prefix="spadiva_split_") as temporary:
            path = Path(temporary) / "split.npz"
            np.savez(path, **self.index_split)
            result = self.prepare(split=path, latent_adata=self.latent[::-1].copy())
            self.assert_expected_alignment(result)

    def test_h5ad_paths_are_read_by_scanpy_and_aligned(self):
        with tempfile.TemporaryDirectory(prefix="spadiva_anndata_") as temporary:
            paths = {}
            for field, adata in (
                ("latent_adata", self.latent[::-1].copy()),
                ("feature_adata", self.features),
                ("external_true_adata", self.truth),
                ("external_pred_adata", self.prediction),
            ):
                paths[field] = Path(temporary) / f"{field}.h5ad"
                adata.write_h5ad(paths[field])
            self.assert_expected_alignment(self.prepare(**paths))

    def test_missing_requested_observations_raise_instead_of_shrinking_split(self):
        for field, adata, missing in (
            ("latent_adata", self.latent, "B"),
            ("latent_adata", self.latent, "G"),
            ("feature_adata", self.features, "B"),
            ("feature_adata", self.features, "G"),
            ("external_true_adata", self.truth, "F"),
            ("external_pred_adata", self.prediction, "F"),
        ):
            with self.subTest(field=field, missing=missing):
                subset = adata[adata.obs_names != missing].copy()
                with self.assertRaises((ValueError, KeyError)):
                    self.prepare(split=self.named_split, **{field: subset})

    def test_missing_unused_observations_are_allowed(self):
        split = {"train_names": list("AB"), "test_names": list("EF")}
        result = self.prepare(
            split=split,
            latent_adata=self.latent[list("FAEB")].copy(),
            external_true_adata=self.truth[list("EF")].copy(),
            external_pred_adata=self.prediction[list("FE")].copy(),
        )
        self.assert_expected_alignment(result, train="AB", test="EF")

    def test_duplicate_observation_names_are_rejected(self):
        for field, adata in (
            ("latent_adata", self.latent),
            ("feature_adata", self.features),
            ("external_true_adata", self.truth),
            ("external_pred_adata", self.prediction),
        ):
            with self.subTest(field=field):
                duplicate = adata.copy()
                names = list(duplicate.obs_names)
                names[1] = names[0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    duplicate.obs_names = names
                    with self.assertRaises(ValueError):
                        self.prepare(split=self.named_split, **{field: duplicate})

    def test_invalid_integer_indices_are_rejected_without_coercion(self):
        invalid = {
            "negative": np.array([-1, 0, 1, 2]),
            "out_of_bounds": np.array([0, 1, 2, 8]),
            "integral_float": np.array([0.0, 1.0, 2.0, 3.0]),
            "fractional_float": np.array([0.0, 1.5, 2.0, 3.0]),
            "boolean": np.array([False, True]),
            "mixed_boolean": [0, True, 2, 3],
            "two_dimensional": np.array([[0, 1], [2, 3]]),
            "duplicate": np.array([0, 1, 1, 3]),
            "nan": np.array([0, 1, 2, np.nan]),
            "infinity": np.array([0, 1, 2, np.inf]),
        }
        for field in ("idx_train", "idx_test"):
            for label, values in invalid.items():
                with self.subTest(field=field, kind=label):
                    split = dict(self.index_split)
                    split[field] = values
                    with self.assertRaises((ValueError, TypeError)) as caught:
                        self.prepare(split=split)
                    # Reject the invalid index field itself, rather than passing
                    # this test because coercion caused a later overlap error.
                    self.assertIn(field, str(caught.exception))

    def test_overlapping_integer_train_and_test_are_rejected(self):
        split = {"idx_train": np.array([0, 1, 4, 5]), "idx_test": np.array([4, 5, 6, 7])}
        with self.assertRaises(ValueError):
            self.prepare(split=split)

    def test_duplicate_overlapping_unknown_and_malformed_names_are_rejected(self):
        cases = (
            {"train_names": list("AABC"), "test_names": list("EFGH")},
            {"train_names": list("ABCD"), "test_names": list("EEGH")},
            {"train_names": list("ABEF"), "test_names": list("EFGH")},
            {"train_names": ["A", "B", "C", "unknown"], "test_names": list("EFGH")},
            {"train_names": list("ABCD"), "test_names": ["E", "F", "G", "unknown"]},
            {"train_names": np.array([["A", "B"], ["C", "D"]]), "test_names": list("EFGH")},
            {"train_names": list("ABCD"), "test_names": "EFGH"},
        )
        for split in cases:
            with self.subTest(split=split):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    self.prepare(split=split)

    def test_incomplete_name_pair_does_not_fall_back_to_indices(self):
        for name in ("train_names", "test_names"):
            with self.subTest(name=name):
                split = dict(self.index_split)
                split[name] = self.named_split[name]
                with self.assertRaises((ValueError, KeyError)):
                    self.prepare(split=split)

    def test_incomplete_index_pair_is_rejected(self):
        for name in ("idx_train", "idx_test"):
            with self.subTest(name=name):
                with self.assertRaises((ValueError, KeyError)):
                    self.prepare(split={name: self.index_split[name]})

    def test_each_partition_requires_at_least_two_observations(self):
        for name in ("train_names", "test_names"):
            for count in (0, 1):
                with self.subTest(name=name, count=count):
                    split = dict(self.named_split)
                    split[name] = split[name][:count]
                    with self.assertRaises(ValueError):
                        self.prepare(split=split)

    def test_missing_or_empty_names_are_rejected(self):
        for missing in (None, np.nan, ""):
            with self.subTest(missing=missing):
                split = {"train_names": ["A", "B", "C", missing], "test_names": list("EFGH")}
                with self.assertRaises(ValueError):
                    self.prepare(split=split)

    def test_max_features_requires_a_positive_integer(self):
        for value in (0, -1, True, np.bool_(False), 1.0, np.nan):
            with self.subTest(max_features=value):
                with self.assertRaises(ValueError):
                    self.prepare(max_features=value)

    def test_empty_common_feature_set_has_a_clear_error(self):
        prediction = self.prediction.copy()
        prediction.var_names = ["other0", "other1", "other2"]
        with self.assertRaises(ValueError) as caught:
            self.prepare(external_pred_adata=prediction)
        self.assertIn("feature", str(caught.exception).lower())

    def test_max_features_one_evaluates_one_feature_end_to_end(self):
        prepared = self.prepare(max_features=1)
        assert_array_equal(prepared["feature_indices"], [0])
        result = evaluate_prepared(prepared, ridge_alpha=1e-4)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.iloc[0]["feature_name"], "f0")
        self.assertEqual(result.iloc[0]["z_source"], "Z_poe")
        self.assertTrue(np.isfinite(result[["r2_z", "r2_w", "r2_zw", "unique_factor_ratio"]]).all().all())
        assert_allclose(result["external_feature_pearson"], [1.0], atol=1e-12)


class SingleFeatureRidgeTests(unittest.TestCase):
    def setUp(self):
        self.train = np.arange(6, dtype=float).reshape(-1, 1)
        self.test = np.arange(6, 12, dtype=float).reshape(-1, 1)
        self.target = 2 * self.train + 3

    def test_two_dimensional_single_target_has_two_dimensional_predictions(self):
        prediction = ridge_predict(self.train, self.target, self.test)
        self.assertEqual(prediction.shape, (6, 1))

    def test_one_dimensional_direct_target_keeps_one_dimensional_predictions(self):
        prediction = ridge_predict(self.train, self.target[:, 0], self.test)
        self.assertEqual(prediction.shape, (6,))
        matrix_prediction = ridge_predict(self.train, self.target, self.test)
        assert_allclose(prediction, matrix_prediction[:, 0])

    def test_multiple_targets_keep_feature_axis(self):
        targets = np.column_stack((self.target, self.target + 5))
        prediction = ridge_predict(self.train, targets, self.test)
        self.assertEqual(prediction.shape, (6, 2))

    def test_single_feature_evaluation_and_partial_missing_external_predictions(self):
        truth = 2 * self.test + 3
        prediction = truth.copy()
        prediction[0, 0] = np.inf
        prediction[-1, 0] = np.nan
        result = evaluate_unique_factor_ratio(
            self.train, np.zeros_like(self.train), self.target,
            self.test, np.zeros_like(self.test), truth,
            external_pred=prediction,
            feature_names=["protein"],
        )
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.iloc[0]["feature_name"], "protein")
        self.assertTrue(np.isfinite(result[["r2_z", "r2_w", "r2_zw"]]).all().all())
        assert_allclose(result["external_feature_pearson"], [1.0], atol=1e-12)


class FeaturewiseStatisticsTests(unittest.TestCase):
    def test_fully_finite_arrays_preserve_previous_formulas(self):
        rng = np.random.default_rng(2048)
        truth = rng.normal(size=(24, 5))
        prediction = truth + rng.normal(scale=0.6, size=truth.shape)
        t = truth - truth.mean(axis=0, keepdims=True)
        p = prediction - prediction.mean(axis=0, keepdims=True)
        previous_pearson = np.sum(t * p, axis=0) / np.sqrt(np.sum(t ** 2, axis=0) * np.sum(p ** 2, axis=0))
        previous_r2 = 1 - np.sum((truth - prediction) ** 2, axis=0) / np.sum(t ** 2, axis=0)
        assert_allclose(featurewise_pearson(truth, prediction), previous_pearson, rtol=1e-12, atol=1e-12)
        assert_allclose(featurewise_r2(truth, prediction), previous_r2, rtol=1e-12, atol=1e-12)

    def test_statistics_use_same_finite_pairs_for_centering_and_errors(self):
        truth = np.column_stack((
            [1, 2, 3, 4, 5, 6],
            [1, np.nan, 3, 4, 5, 6],
            [np.inf, 2, 3, -np.inf, 5, 6],
        ))
        prediction = np.column_stack((
            [1, 2, np.nan, 4, np.inf, 6],
            [2, 10, 6, 8, np.nan, 12],
            [4, np.nan, 6, 8, 10, np.nan],
        ))
        expected_pearson, expected_r2 = [], []
        for column in range(truth.shape[1]):
            paired = np.isfinite(truth[:, column]) & np.isfinite(prediction[:, column])
            t, p = truth[paired, column], prediction[paired, column]
            expected_pearson.append(pearsonr(t, p).statistic)
            expected_r2.append(r2_score(t, p, force_finite=False))
        assert_allclose(featurewise_pearson(truth, prediction), expected_pearson, atol=1e-12)
        assert_allclose(featurewise_r2(truth, prediction), expected_r2, atol=1e-12)

    def test_undefined_statistics_return_nan_not_perfect_scores(self):
        cases = (
            ("all_prediction_nan", [1, 2, 3, 4], [np.nan] * 4),
            ("all_prediction_inf", [1, 2, 3, 4], [np.inf, -np.inf, np.inf, -np.inf]),
            ("all_truth_nan", [np.nan] * 4, [1, 2, 3, 4]),
            ("one_finite_pair", [1, np.nan, 3, np.inf], [2, 3, np.nan, 5]),
            ("no_finite_pairs", [np.nan, 2, np.inf, 4], [1, np.nan, 3, np.inf]),
            ("constant_truth", [2, 2, 2, 2], [1, 2, 3, 4]),
            ("constant_after_pair_filter", [2, 2, 3, 4], [1, 3, np.nan, np.inf]),
        )
        for label, truth, prediction in cases:
            with self.subTest(case=label):
                truth = np.asarray(truth, dtype=float).reshape(-1, 1)
                prediction = np.asarray(prediction, dtype=float).reshape(-1, 1)
                self.assertTrue(np.isnan(featurewise_pearson(truth, prediction)[0]))
                self.assertTrue(np.isnan(featurewise_r2(truth, prediction)[0]))

    def test_constant_prediction_has_nan_pearson_but_valid_r2(self):
        truth = np.arange(1, 7, dtype=float).reshape(-1, 1)
        prediction = np.full_like(truth, truth.mean())
        self.assertTrue(np.isnan(featurewise_pearson(truth, prediction)[0]))
        assert_allclose(featurewise_r2(truth, prediction), [0.0], atol=1e-12)
        negative = featurewise_r2(truth, np.zeros_like(truth))[0]
        self.assertLess(negative, 0)

    def test_one_dimensional_single_feature_inputs_are_supported(self):
        truth = np.arange(1, 7, dtype=float)
        prediction = truth + 0.5
        self.assertEqual(featurewise_pearson(truth, prediction).shape, (1,))
        self.assertEqual(featurewise_r2(truth, prediction).shape, (1,))
        assert_allclose(featurewise_pearson(truth, prediction), [1.0], atol=1e-12)
        assert_allclose(featurewise_r2(truth, prediction), [r2_score(truth, prediction)], atol=1e-12)

    def test_shape_mismatch_is_rejected(self):
        for statistic in (featurewise_pearson, featurewise_r2):
            with self.subTest(statistic=statistic.__name__):
                with self.assertRaises(ValueError):
                    statistic(np.ones((4, 2)), np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
