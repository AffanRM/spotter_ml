"""Verify ID alignment, fixed chart inputs, and native model persistence."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from freight.policy import ROOT, load_policy, make_policy_model
from freight.submission import december_predictions, load_bundle, save_bundle, validation_predictions
from test_data import sample_loads


class DistanceModel:
    def predict(self, frame):
        return frame.distance.to_numpy(dtype=float)


class SubmissionTests(unittest.TestCase):
    def test_final_predictions_join_by_id_and_preserve_template_order(self):
        ids = [f"TE-{i:06d}" for i in range(1, 12001)]
        loads = pd.DataFrame({"load_id": ids, "distance": np.arange(1, 12001)})
        template = pd.DataFrame({"load_id": ids[::-1], "predicted_rate": np.nan})
        result = validation_predictions(DistanceModel(), loads, template)
        self.assertEqual(result.load_id.tolist(), ids[::-1])
        self.assertEqual(result.predicted_rate.iloc[0], 12000.)
        self.assertEqual(result.predicted_rate.iloc[-1], 1.)
        self.assertTrue(template.predicted_rate.isna().all())
        with self.assertRaises(ValueError):
            validation_predictions(DistanceModel(), loads.iloc[:-1], template)
        duplicate = template.copy()
        duplicate.loc[1, "load_id"] = duplicate.loc[0, "load_id"]
        with self.assertRaises(ValueError):
            validation_predictions(DistanceModel(), loads, duplicate)

    def test_december_preserves_every_input_and_original_row_order(self):
        inputs = pd.DataFrame({
            "pickup": ["Lexington"] * 31, "delivery": ["Fort Wayne"] * 31,
            "distance": [360] * 31, "equipment": ["Dry Van"] * 31,
            "weight": [32000] * 31,
            "date": pd.date_range("2025-12-01", "2025-12-31").strftime("%Y-%m-%d"),
            "predicted_rate": np.nan,
        }).sample(frac=1, random_state=42)
        original = inputs.copy(deep=True)
        completed = december_predictions(DistanceModel(), inputs)
        pd.testing.assert_frame_equal(inputs, original)
        pd.testing.assert_frame_equal(completed.drop(columns="predicted_rate"), original.drop(columns="predicted_rate"))
        self.assertTrue(completed.predicted_rate.eq(360).all())
        with self.assertRaises(ValueError):
            december_predictions(DistanceModel(), inputs.drop(columns="predicted_rate"))
        invalid = inputs.copy()
        invalid.loc[invalid.index[0], "weight"] = 31000
        with self.assertRaises(SystemExit):
            december_predictions(DistanceModel(), invalid)

    def test_native_models_reload_exact_predictions_and_detect_tampering(self):
        train = sample_loads()
        train["date"] = pd.to_datetime(train.date)
        policy = load_policy()
        policy["iterations"] = 5  # Small fixture; production policy is unchanged.
        models = {role: make_policy_model(policy, role).fit(train) for role in ["main", "chart"]}
        identity = {"experiment_id": "test", "input_sha256": "test-input", "implementation_sha256": {}, "python_version": "test", "packages": {}}
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            directory = Path(directory)
            save_bundle(models, train, policy, identity, directory)
            reloaded, _ = load_bundle(directory, policy, identity)
            for role in models:
                np.testing.assert_allclose(models[role].predict(train), reloaded[role].predict(train), rtol=0, atol=1e-10)
            wrong_identity = {**identity, "experiment_id": "different"}
            with self.assertRaisesRegex(ValueError, "different input"):
                load_bundle(directory, policy, wrong_identity)
            metadata_path = directory / "manifest.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["models"]["main"]["sha256"] = "incorrect-checksum"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_bundle(directory, policy, identity)


if __name__ == "__main__":
    unittest.main()
