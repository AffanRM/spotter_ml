"""Guard against silently changing inputs, invalid labels, and ID leakage."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from freight.data import FEATURE_COLUMNS, clean_features, read_loads


def sample_loads():
    return pd.DataFrame({
        "load_id": ["A", "B", "C"],
        "pickup": ["Lexington"] * 3,
        "delivery": ["Fort Wayne"] * 3,
        "pickup_lat": [38.] * 3, "pickup_lon": [-84.] * 3,
        "delivery_lat": [41.] * 3, "delivery_lon": [-85.] * 3,
        "distance": [360.] * 3, "equipment": ["Dry Van"] * 3,
        "weight": [32000., -30000., np.nan],
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "market_index": [1., np.nan, 1.1], "quote_signal": [2.] * 3,
        "posted_rate": [700., 750., 800.],
    })


class CleaningTests(unittest.TestCase):
    def test_unknown_weights_are_flagged_without_mutating_raw_inputs(self):
        raw = sample_loads()
        original = raw.copy(deep=True)
        cleaned = clean_features(raw)
        pd.testing.assert_frame_equal(raw, original)
        self.assertEqual(cleaned.weight.iloc[0], 32000.)
        self.assertTrue(cleaned.weight.iloc[1:].isna().all())
        self.assertEqual(cleaned.weight_missing.tolist(), [0, 0, 1])
        self.assertEqual(cleaned.weight_invalid.tolist(), [0, 1, 0])
        self.assertEqual(cleaned.market_index_missing.tolist(), [0, 1, 0])
        self.assertNotIn("load_id", cleaned)
        self.assertNotIn("posted_rate", cleaned)

    def test_zero_weight_is_unknown_and_inference_needs_no_labels(self):
        raw = sample_loads()[FEATURE_COLUMNS].copy()
        raw.loc[0, "weight"] = 0
        cleaned = clean_features(raw)
        self.assertTrue(pd.isna(cleaned.weight.iloc[0]))
        self.assertEqual(cleaned.weight_invalid.iloc[0], 1)
        self.assertEqual(len(cleaned), len(raw))

    def test_cleaning_does_not_learn_from_other_rows(self):
        raw = sample_loads()
        alone = clean_features(raw.iloc[[2]])
        together = clean_features(raw)
        pd.testing.assert_frame_equal(alone, together.iloc[[2]])


class InputContractTests(unittest.TestCase):
    def read_fixture(self, frame):
        scratch = Path(__file__).resolve().parents[1] / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "loads.csv"
            frame.to_csv(path, index=False)
            return read_loads(path, labeled=True)

    def test_missing_values_and_negative_weights_remain_auditable(self):
        frame = self.read_fixture(sample_loads())
        self.assertEqual(frame.weight.iloc[1], -30000.)
        self.assertTrue(pd.isna(frame.weight.iloc[2]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame.date))

    def test_invalid_inputs_fail_instead_of_disappearing(self):
        for column, value in [
            ("posted_rate", 0), ("posted_rate", np.nan),
            ("posted_rate", np.inf), ("distance", -1),
            ("pickup_lat", 91), ("date", "invalid-date"),
            ("pickup", " "), ("quote_signal", "invalid-number"),
        ]:
            with self.subTest(column=column, value=value):
                frame = sample_loads()
                frame[column] = frame[column].astype(object)
                frame.loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.read_fixture(frame)

    def test_duplicate_ids_are_rejected(self):
        frame = sample_loads()
        frame.loc[1, "load_id"] = frame.loc[0, "load_id"]
        with self.assertRaisesRegex(ValueError, "duplicate load_id"):
            self.read_fixture(frame)


if __name__ == "__main__":
    unittest.main()
