"""Protect city holdout isolation and provenance of residual diagnostics."""

import unittest

import pandas as pd

from freight.models import MODEL_SPECS
from freight.robustness import error_summary, validate_development_predictions
from freight.splits import choose_heldout_cities, purge_cities


class CityHoldoutTests(unittest.TestCase):
    def test_selection_is_independent_of_targets_and_row_order(self):
        frame = pd.DataFrame({
            "pickup": ["A", "B", "C", "D"], "delivery": ["B", "C", "D", "A"],
            "posted_rate": [1., 2., 3., 4.],
        })
        changed = frame.sample(frac=1, random_state=1).copy()
        changed["posted_rate"] = 1000000
        self.assertEqual(choose_heldout_cities(frame, count=2), choose_heldout_cities(changed, count=2))
        with self.assertRaises(ValueError):
            choose_heldout_cities(frame, count=4)

    def test_purge_removes_both_pickup_and_delivery_roles(self):
        frame = pd.DataFrame({"pickup": ["A", "B", "C"], "delivery": ["B", "A", "B"]})
        original = frame.copy()
        retained = purge_cities(frame, ["A"])
        self.assertEqual(retained.index.tolist(), [2])
        pd.testing.assert_frame_equal(frame, original)
        with self.assertRaises(ValueError):
            purge_cities(frame, ["A", "B", "C"])


class PredictionAuditTests(unittest.TestCase):
    def fixture(self):
        development = pd.DataFrame({
            "load_id": ["past", "may", "july"],
            "date": pd.to_datetime(["2025-01-01", "2025-05-01", "2025-07-01"]),
            "posted_rate": [100., 200., 300.],
        })
        oof = pd.DataFrame([
            {"fold": fold, "model": spec.name, "load_id": load_id, "actual_rate": actual, "predicted_rate": actual + 1}
            for spec in MODEL_SPECS
            for fold, load_id, actual in [("development_1", "may", 200.), ("development_2", "july", 300.)]
        ])
        return development, oof

    def test_complete_predictions_are_accepted(self):
        source, oof = self.fixture()
        validate_development_predictions(oof, source)

    def test_final_check_ids_stale_labels_and_duplicates_are_rejected(self):
        source, oof = self.fixture()
        contaminated = oof.copy()
        contaminated.loc[0, "load_id"] = "september-final-check"
        stale = oof.copy()
        stale.loc[0, "actual_rate"] = 999
        wrong_fold = oof.copy()
        wrong_fold.loc[0, ["load_id", "actual_rate"]] = ["july", 300.]
        for invalid in [contaminated, stale, wrong_fold, pd.concat([oof, oof.iloc[[0]]]), oof.iloc[1:]]:
            with self.subTest(rows=len(invalid)):
                with self.assertRaises(ValueError):
                    validate_development_predictions(invalid, source)

    def test_error_concentration_keeps_large_errors_in_headline_metrics(self):
        group = pd.DataFrame({"actual_rate": [10., 10., 10., 10.], "predicted_rate": [10., 10., 10., 14.]})
        result = error_summary(group)
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["mae"], 1.)
        self.assertEqual(result["rmse"], 2.)
        self.assertEqual(result["largest_1pct_squared_error_share"], 1.)


if __name__ == "__main__":
    unittest.main()
