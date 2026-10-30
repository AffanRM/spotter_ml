"""Check inference contracts, leakage boundaries, and unfamiliar categories."""

import unittest

import numpy as np
import pandas as pd

from freight.features import build_features
from freight.models import ModelSpec, RateModel
from test_data import sample_loads


class FeatureTests(unittest.TestCase):
    def test_chart_features_match_when_omitted_signals_are_unavailable(self):
        full = sample_loads()
        reduced = full[["pickup", "delivery", "distance", "equipment", "weight", "date"]]
        pd.testing.assert_frame_equal(build_features(full, "chart"), build_features(reduced, "chart"))

    def test_core_is_independent_of_signals_ids_and_labels(self):
        full = sample_loads()
        changed = full.copy()
        changed["quote_signal"] = 999999
        changed["market_index"] = np.nan
        changed["load_id"] = "different"
        changed["posted_rate"] = 999999
        features = build_features(full, "core")
        pd.testing.assert_frame_equal(features, build_features(changed, "core"))
        self.assertNotIn("load_id", features)
        self.assertNotIn("posted_rate", features)

    def test_full_model_cannot_silently_use_chart_inputs(self):
        reduced = sample_loads().drop(columns="quote_signal")
        with self.assertRaisesRegex(ValueError, "quote_signal"):
            build_features(reduced, "full")


class ModelTests(unittest.TestCase):
    def test_equipment_baseline_falls_back_to_training_global_rate(self):
        train = sample_loads()
        model = RateModel(ModelSpec("baseline", "equipment_rpm")).fit(train)
        future = train.iloc[[0]].copy()
        future["equipment"] = "Unseen equipment"
        future["distance"] = 720
        self.assertAlmostEqual(model.predict(future)[0], 1500.)

    def test_ridge_imputation_uses_training_values_and_handles_unseen_cities(self):
        train = sample_loads()
        model = RateModel(ModelSpec("ridge", "ridge")).fit(train)
        numeric = model.estimator_.steps[0][1].named_transformers_["numeric"]
        imputer = numeric.steps[0][1]
        before = imputer.statistics_.copy()
        future = train.iloc[[2]].drop(columns="posted_rate").copy()
        future["pickup"] = "New city"
        future["weight"] = 1000000
        prediction = model.predict(future)
        np.testing.assert_equal(imputer.statistics_, before)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())

    def test_chart_model_predicts_without_target_signals_or_coordinates(self):
        train = sample_loads()
        model = RateModel(ModelSpec("chart", "catboost", "chart"), iterations=5, threads=1).fit(train)
        future = train[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
        future["pickup"] = "New city"
        prediction = model.predict(future)
        self.assertEqual(prediction.shape, (3,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())


if __name__ == "__main__":
    unittest.main()
