"""Check the frozen policy and preservation of previous evaluation evidence."""

import json
from pathlib import Path
import tempfile
import unittest

from freight.final_check import experiment_identity, run
from freight.policy import ROOT, load_policy, make_policy_model


class PolicyTests(unittest.TestCase):
    def test_declared_roles_use_expected_features_and_fixed_parameters(self):
        policy = load_policy()
        self.assertEqual(make_policy_model(policy, "main").spec.feature_set, "full")
        self.assertEqual(make_policy_model(policy, "chart").spec.feature_set, "chart")
        self.assertEqual(make_policy_model(policy, "reference").spec.kind, "equipment_rpm")
        for role in ["main", "chart"]:
            model = make_policy_model(policy, role)
            self.assertEqual(model.iterations, 500)
            self.assertEqual(model.seed, 42)
            self.assertEqual(model.spec.loss, "MAE")
        with self.assertRaises(ValueError):
            make_policy_model(policy, "choose_after_evaluation")

    def test_invalid_or_unimplemented_policies_are_rejected(self):
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "policy.json"
            for key, invalid in [
                ("policy_version", 99), ("main_model", "unknown"),
                ("chart_model", "catboost_full_mae"), ("main_model", "catboost_chart_mae"),
                ("reference_model", "median_rate"), ("iterations", 0),
                ("threads", True), ("seed", "42"),
                ("unseen_city_routing", True), ("drop_target_outliers", True),
                ("train_end", "2025-10-31"),
            ]:
                with self.subTest(key=key, value=invalid):
                    policy = load_policy()
                    policy[key] = invalid
                    path.write_text(json.dumps(policy), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_policy(path)

    def test_changed_experiment_cannot_overwrite_existing_final_results(self):
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            directory = Path(directory)
            data = directory / "loads.csv"
            data.write_text("test input whose hash is tracked", encoding="utf-8")
            policy_path = ROOT / "model_policy.json"
            identity = experiment_identity(data, policy_path)
            self.assertEqual(identity, experiment_identity(data, policy_path))
            output = directory / "existing"
            output.mkdir()
            manifest = output / "manifest.json"
            manifest.write_text(json.dumps({"experiment_id": "prior-experiment"}), encoding="utf-8")
            original = manifest.read_bytes()
            with self.assertRaisesRegex(ValueError, "different policy"):
                run(data, policy_path, output)
            self.assertEqual(manifest.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
