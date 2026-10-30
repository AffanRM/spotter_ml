"""Independently verify published deliverables, provenance, and local models."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import score
from freight.benchmark import metrics
from freight.data import read_loads
from freight.final_check import experiment_identity
from freight.policy import ROOT, load_policy
from freight.robustness import validate_development_predictions
from freight.submission import load_bundle, sha256, verify_evaluation_provenance


def audit(output: Path, require_saved_models: bool = False) -> list[dict]:
    results = []

    def check(name, action):
        try:
            detail = action()
            results.append({"check": name, "status": "pass", "detail": detail or "Verified"})
        except (Exception, SystemExit) as error:
            results.append({"check": name, "status": "fail", "detail": str(error)})

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    train = read_loads(ROOT / "train-test.csv", labeled=True)
    final = read_loads(ROOT / "validation.csv", labeled=False)
    template = pd.read_csv(ROOT / "validation-predictions-template.csv")
    predictions = pd.read_csv(ROOT / "validation_predictions.csv")
    december_input = pd.read_csv(ROOT / "december-chart-inputs.csv")
    december = pd.read_csv(ROOT / "december_predictions.csv")
    submission = json.loads((ROOT / "reports/submission/manifest.json").read_text())
    evaluation = json.loads((ROOT / "reports/final_check/manifest.json").read_text())
    identity = experiment_identity(ROOT / "train-test.csv", ROOT / "model_policy.json")

    def final_contract():
        score.validate_predictions(predictions)
        require(predictions.load_id.tolist() == template.load_id.tolist(), "Template order changed")
        require(set(predictions.load_id) == set(final.load_id), "Final IDs differ")
        return "12,000 unique IDs; exact two-column schema; finite positive rates; template order"

    def december_contract():
        score.validate_december(december)
        pd.testing.assert_frame_equal(december_input.drop(columns="predicted_rate"), december.drop(columns="predicted_rate"))
        return "31 dates; fixed route and inputs preserved; positive finite predictions"

    def input_integrity():
        initial = json.loads((ROOT / "reports/eda/audit.json").read_text())
        for name, digest in initial["input_sha256"].items():
            require(sha256(ROOT / name) == digest, f"Raw input changed: {name}")
        require(sha256(ROOT / "score.py") == submission["scorer_sha256"], "Scorer changed")
        return "Four original CSV hashes and supplied scorer match recorded provenance"

    def output_integrity():
        for name, digest in submission["output_sha256"].items():
            require(sha256(ROOT / name) == digest, f"Output hash mismatch: {name}")
        require(sha256(ROOT / "freight/submission.py") == submission["generation_implementation_sha256"], "Generation audit is stale")
        return "Both CSVs, chart, and generation implementation match their audit"

    def provenance():
        verify_evaluation_provenance(identity, evaluation)
        require(submission["frozen_experiment_id"] == evaluation["experiment_id"], "Wrong evaluated policy reference")
        require(submission["training_rows"] == len(train) == 48000, "Wrong refit row count")
        require(submission["training_date_max"] == str(train.date.max().date()), "Wrong refit date range")
        return "Frozen data/code/policy preserved; full-data refit covers all 48,000 labeled rows"

    def split_boundaries():
        require(train.date.max() < final.date.min(), "Final dates overlap labeled dates")
        require(not set(train.load_id) & set(final.load_id), "ID overlap")
        require(evaluation["train_date_max"] < evaluation["validation_date_min"], "Evaluation dates overlap")
        require(evaluation["train_rows"] == 38477 and evaluation["validation_rows"] == 9523, "Evaluation counts changed")
        return "Development, reserved evaluation, and final inference boundaries checked"

    def report_metrics():
        table = pd.read_csv(ROOT / "reports/final_check/metrics.csv").set_index("role")
        require(set(table.index) == {"reference", "main", "chart"} and len(table) == 3, "Wrong evaluation metric roles")
        require(np.isfinite(table[["mae", "rmse", "r2"]].to_numpy()).all(), "Invalid stored metric values")
        metric_path = ROOT / "artifacts/final_check/predictions.csv"
        if not metric_path.exists():
            return "Stored metric table present; row-level recomputation requires optional local evaluation artifacts"
        rows = pd.read_csv(metric_path)
        labels = train.set_index("load_id").posted_rate
        expected_ids = set(train.loc[train.date.between("2025-09-01", "2025-10-31"), "load_id"])
        require(set(rows.role) == {"reference", "main", "chart"}, "Missing evaluation role")
        for role, group in rows.groupby("role"):
            require(len(group) == 9523 and not group.load_id.duplicated().any() and set(group.load_id) == expected_ids, "Wrong evaluation IDs")
            np.testing.assert_allclose(group.actual_rate, group.load_id.map(labels), rtol=0, atol=1e-8)
            calculated = metrics(group.actual_rate, group.predicted_rate)
            np.testing.assert_allclose([calculated[k] for k in ["mae", "rmse", "r2"]], table.loc[role, ["mae", "rmse", "r2"]].to_numpy(dtype=float), rtol=1e-10)
        return "All three reserved-window metric rows independently recomputed from 9,523 labeled predictions each"

    def development_metrics():
        table = pd.read_csv(ROOT / "reports/benchmark/leaderboard.csv").set_index("model")
        require(len(table) == 8 and np.isfinite(table[["mae", "rmse", "r2"]].to_numpy()).all(), "Invalid development leaderboard")
        path = ROOT / "artifacts/benchmark/development_predictions.csv"
        if not path.exists():
            return "Eight stored model results checked; optional prediction artifacts absent on this checkout"
        rows = pd.read_csv(path)
        validate_development_predictions(rows, train.loc[train.date.le("2025-08-31")])
        for model, group in rows.groupby("model"):
            calculated = metrics(group.actual_rate, group.predicted_rate)
            np.testing.assert_allclose([calculated[k] for k in ["mae", "rmse", "r2"]], table.loc[model, ["mae", "rmse", "r2"]].to_numpy(dtype=float), rtol=1e-10)
        return "All eight development leader-board rows recomputed; labels, folds, and prediction IDs checked"

    def city_metrics():
        table = pd.read_csv(ROOT / "reports/robustness/location_pooled_metrics.csv").set_index(["model", "segment"])
        require(len(table) == 10 and np.isfinite(table[["mae", "rmse", "r2"]].to_numpy()).all(), "Invalid city stress table")
        path = ROOT / "artifacts/robustness/city_holdout_predictions.csv"
        if not path.exists():
            return "Ten stored city-stress results checked; optional prediction artifacts absent on this checkout"
        rows = pd.read_csv(path)
        require(not rows.duplicated(["fold", "model", "load_id"]).any(), "Duplicate city-stress predictions")
        labels = train.set_index("load_id").posted_rate
        np.testing.assert_allclose(rows.actual_rate, rows.load_id.map(labels), rtol=0, atol=1e-8)
        require(set(rows.groupby(["model", "segment"]).groups) == set(table.index), "City stress result coverage mismatch")
        for key, group in rows.groupby(["model", "segment"]):
            calculated = metrics(group.actual_rate, group.predicted_rate)
            np.testing.assert_allclose([calculated[k] for k in ["rows", "mae", "rmse", "r2"]], table.loc[key, ["rows", "mae", "rmse", "r2"]].to_numpy(dtype=float), rtol=1e-10)
        return "All ten pooled seen/unseen city metrics independently recomputed with source labels"

    def local_links():
        documents = list(ROOT.glob("*.md")) + list((ROOT / "reports").rglob("*.md"))
        for document in documents:
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                path = (document.parent / target.split("#")[0]).resolve()
                # This command writes its own result after all checks finish.
                if path == (output / "checks.json").resolve():
                    continue
                require(path.exists(), f"Broken link in {document.name}: {target}")
        return f"Relative file links checked in {len(documents)} Markdown documents"

    check("Final prediction contract", final_contract)
    check("December prediction contract", december_contract)
    check("Original input and scorer integrity", input_integrity)
    check("Published output integrity", output_integrity)
    check("Frozen evaluation and refit provenance", provenance)
    check("Temporal and ID separation", split_boundaries)
    check("Reserved metric arithmetic", report_metrics)
    check("Development metric arithmetic", development_metrics)
    check("City stress metric arithmetic", city_metrics)
    check("Documentation links", local_links)
    bundle_dir = ROOT / "artifacts/models"
    if (bundle_dir / "manifest.json").exists():
        def saved_predictions():
            models, bundle = load_bundle(bundle_dir, load_policy(), identity)
            require(bundle["train_rows"] == len(train), "Saved training count mismatch")
            main = pd.Series(models["main"].predict(final), index=final.load_id)
            np.testing.assert_allclose(predictions.predicted_rate, predictions.load_id.map(main), rtol=0, atol=5.1e-7)
            np.testing.assert_allclose(december.predicted_rate, models["chart"].predict(december_input.drop(columns="predicted_rate")), rtol=0, atol=5.1e-7)
            require(bundle.get("reload_verified_on_all_submission_rows") is True, "Original save/reload verification not recorded; run the refit command")
            return "Checked model hashes, full training count, reload evidence, and every written prediction"
        check("Saved models and prediction agreement", saved_predictions)
    else:
        results.append({"check": "Saved models and prediction agreement", "status": "fail" if require_saved_models else "optional", "detail": "Run python -m freight.submission to regenerate ignored model artifacts"})
    output.mkdir(parents=True, exist_ok=True)
    report = {"technical_checks_passed": not any(r["status"] == "fail" for r in results), "checks": results}
    (output / "checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for row in results:
        print(f"{row['status'].upper()}: {row['check']}: {row['detail']}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/audit")
    parser.add_argument("--require-saved-models", action="store_true")
    args = parser.parse_args()
    results = audit(args.output_dir.resolve(), args.require_saved_models)
    if any(row["status"] == "fail" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
