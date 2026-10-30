"""Refit frozen models and generate validated submission files.

Run python -m freight.submission, or add --predict-only to reuse saved models.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

import score as scorer
from freight.data import TARGET, read_loads
from freight.final_check import experiment_identity
from freight.features import build_features
from freight.policy import ROOT, load_policy, make_policy_model


INPUT_FILES = ["train-test.csv", "validation.csv", "validation-predictions-template.csv", "december-chart-inputs.csv"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_evaluation_provenance(identity: dict, evaluated: dict) -> bool:
    """Keep evaluated data/code/policy fixed; record environment differences.

    Reproduction on another Python patch release must not require overwriting
    the original holdout report. Its metrics still describe its own environment.
    Saved model bundles remain tied to the environment in which they were fit.
    """
    if not evaluated.get("final_check_evaluated"):
        raise ValueError("The frozen policy has not been evaluated")
    for key in ["input_sha256", "policy_sha256", "implementation_sha256"]:
        if identity.get(key) != evaluated.get(key):
            raise ValueError(f"Input, policy, or evaluated implementation changed: {key}")
    return all(identity.get(key) == evaluated.get(key) for key in ["python_version", "packages"])


def validation_predictions(model, loads: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """Fill by ID, preserving template order rather than assuming row alignment."""
    if list(template.columns) != ["load_id", "predicted_rate"]:
        raise ValueError("Unexpected prediction template columns")
    for name, frame in [("loads", loads), ("template", template)]:
        if frame.load_id.isna().any() or frame.load_id.duplicated().any():
            raise ValueError(f"Missing or duplicate IDs in {name}")
    if len(loads) != len(template) or set(loads.load_id) != set(template.load_id):
        raise ValueError("Final load IDs and template IDs must match exactly")
    predicted = pd.Series(model.predict(loads), index=loads.load_id)
    result = template.copy()
    result["predicted_rate"] = result.load_id.map(predicted)
    scorer.validate_predictions(result)
    return result


def december_predictions(model, inputs: pd.DataFrame) -> pd.DataFrame:
    """Validate fixed inputs and fill only the prediction column."""
    expected = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if list(inputs.columns) != expected:
        raise ValueError("December inputs must preserve the original seven columns")
    result = inputs.copy()
    # Validate fixed fields before model inference; predictions are not yet known.
    candidate = result.copy()
    candidate["predicted_rate"] = 1.
    scorer.validate_december(candidate)
    result["predicted_rate"] = model.predict(result.drop(columns="predicted_rate"))
    scorer.validate_december(result)
    return result


def save_bundle(models: dict, train: pd.DataFrame, policy: dict, identity: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "bundle_version": 1, "frozen_experiment_id": identity["experiment_id"],
        "training_input_sha256": identity["input_sha256"],
        "policy": policy, "implementation_sha256": identity["implementation_sha256"],
        "python_version": identity["python_version"], "packages": identity["packages"],
        "train_rows": len(train), "train_date_min": str(train.date.min().date()),
        "train_date_max": str(train.date.max().date()), "models": {},
    }
    for role in ["main", "chart"]:
        path = directory / f"{role}.cbm"
        models[role].estimator_.save_model(str(path), format="cbm")
        metadata["models"][role] = {
            "filename": path.name, "sha256": sha256(path),
            "configuration": models[role].configuration(),
            "features": build_features(train, models[role].spec.feature_set).columns.tolist(),
        }
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_bundle(directory: Path, policy: dict, identity: dict) -> tuple[dict, dict]:
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("bundle_version") != 1 or metadata.get("policy") != policy:
        raise ValueError("Saved models do not match the frozen policy")
    if metadata.get("frozen_experiment_id") != identity["experiment_id"]:
        raise ValueError("Saved models belong to a different input, implementation, or environment")
    if set(metadata.get("models", {})) != {"main", "chart"}:
        raise ValueError("Saved bundle must contain main and chart models")
    models = {}
    for role in ["main", "chart"]:
        entry = metadata["models"][role]
        if entry["filename"] != f"{role}.cbm":
            raise ValueError("Unexpected saved model filename")
        path = directory / entry["filename"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Saved {role} model checksum mismatch")
        model = make_policy_model(policy, role)
        if entry["configuration"] != model.configuration():
            raise ValueError(f"Saved {role} configuration mismatch")
        model.estimator_ = CatBoostRegressor()
        model.estimator_.load_model(str(path), format="cbm")
        if model.estimator_.feature_names_ != entry["features"]:
            raise ValueError(f"Saved {role} features mismatch")
        models[role] = model
    return models, metadata


def run(data_dir: Path, policy_path: Path, model_dir: Path, output_dir: Path, predict_only: bool):
    policy = load_policy(policy_path)
    identity = experiment_identity(data_dir / "train-test.csv", policy_path)
    final_manifest = json.loads((ROOT / "reports" / "final_check" / "manifest.json").read_text(encoding="utf-8"))
    same_environment = verify_evaluation_provenance(identity, final_manifest)
    if not same_environment:
        print("Reproducing the frozen configuration in a different environment; original evaluation evidence is preserved.", flush=True)
    input_hashes = {name: sha256(data_dir / name) for name in INPUT_FILES}
    train = read_loads(data_dir / "train-test.csv", labeled=True)
    loads = read_loads(data_dir / "validation.csv", labeled=False)
    template = pd.read_csv(data_dir / "validation-predictions-template.csv")
    december = pd.read_csv(data_dir / "december-chart-inputs.csv")
    if len(train) != 48000 or len(loads) != 12000:
        raise ValueError("Expected 48,000 labeled loads and 12,000 final loads")
    if train.date.max() >= loads.date.min() or set(train.load_id) & set(loads.load_id):
        raise ValueError("Labeled and final loads must be disjoint, with final loads later")
    if predict_only:
        print("Loading saved full-data models; no retraining.", flush=True)
        models, bundle = load_bundle(model_dir, policy, identity)
        reload_verified = bool(bundle.get("reload_verified_on_all_submission_rows", False))
    else:
        models = {}
        for role in ["main", "chart"]:
            print(f"Fitting frozen {role} model on all {len(train):,} labeled loads...", flush=True)
            models[role] = make_policy_model(policy, role).fit(train)
        save_bundle(models, train, policy, identity, model_dir)
        reloaded, bundle = load_bundle(model_dir, policy, identity)
        for role, sample in [("main", loads), ("chart", december.drop(columns="predicted_rate"))]:
            np.testing.assert_allclose(models[role].predict(sample), reloaded[role].predict(sample), rtol=0, atol=1e-10)
        models = reloaded
        bundle["reload_verified_on_all_submission_rows"] = True
        (model_dir / "manifest.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8", newline="\n")
        reload_verified = True
        print("Saved-model reload predictions verified for every final and December row.", flush=True)
    if bundle["train_rows"] != len(train) or bundle["training_input_sha256"] != input_hashes["train-test.csv"]:
        raise ValueError("Saved models were not fitted on the expected full training data")
    final = validation_predictions(models["main"], loads.drop(columns=TARGET, errors="ignore"), template)
    chart = december_predictions(models["chart"], december)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path, december_path = output_dir / "validation_predictions.csv", output_dir / "december_predictions.csv"
    final.to_csv(final_path, index=False, float_format="%.6f", lineterminator="\n")
    chart.to_csv(december_path, index=False, float_format="%.6f", lineterminator="\n")
    # Exercise the exact supplied CLI against the written files, not just frames.
    scorer_dir = output_dir / "scorer_results"
    command = [sys.executable, str(ROOT / "score.py"), "--predictions", str(final_path), "--december-predictions", str(december_path), "--output-dir", str(scorer_dir)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    print(completed.stdout, end="", flush=True)
    if {name: sha256(data_dir / name) for name in INPUT_FILES} != input_hashes:
        raise ValueError("Raw input hashes changed during prediction generation")
    # Check serialized precision and template order as independent write checks.
    written = pd.read_csv(final_path)
    if written.load_id.tolist() != template.load_id.tolist():
        raise ValueError("Written predictions do not preserve template order")
    np.testing.assert_allclose(written.predicted_rate, final.predicted_rate, rtol=0, atol=5.1e-7)
    chart_path = scorer_dir / "candidate_december.png"
    report_dir = output_dir / "reports" / "submission"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "frozen_experiment_id": final_manifest["experiment_id"], "policy": policy,
        "generation_experiment_id": identity["experiment_id"],
        "generation_environment_matches_evaluation": same_environment,
        "generation_environment": {"python_version": identity["python_version"], "packages": identity["packages"]},
        "generation_implementation_sha256": sha256(Path(__file__)),
        "input_sha256": input_hashes, "scorer_sha256": sha256(ROOT / "score.py"),
        "training_rows": len(train), "training_date_min": str(train.date.min().date()),
        "training_date_max": str(train.date.max().date()), "final_rows": len(final), "december_rows": len(chart),
        "all_predictions_finite_positive": True, "template_order_preserved": True,
        "raw_inputs_unchanged": True, "provided_scorer_passed": True,
        "saved_model_reload_verified": reload_verified,
        "output_sha256": {"validation_predictions.csv": sha256(final_path), "december_predictions.csv": sha256(december_path), "scorer_results/candidate_december.png": sha256(chart_path)},
        "models": bundle["models"],
        "final_rate_summary": final.predicted_rate.describe().to_dict(),
        "december_rate_summary": chart.predicted_rate.describe().to_dict(),
        "accuracy_note": "No final labels are available; scorer validates format and chart, not accuracy.",
    }
    (report_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (report_dir / "scorer_output.txt").write_text(completed.stdout, encoding="utf-8", newline="\n")
    text = f"""# Submission generation checkpoint

The frozen main and chart configurations were refitted on **{len(train):,}**
labeled loads, {train.date.min().date()} through {train.date.max().date()}.
No final-set labels exist or were used. No model choices changed after the
September-October evaluation.

## Generated files

- `validation_predictions.csv`: exactly **{len(final):,}** unique final IDs in
  template order and exactly `load_id,predicted_rate` columns.
- `december_predictions.csv`: **{len(chart)}** fixed December loads, with all six
  input fields preserved and only `predicted_rate` filled.
- `scorer_results/candidate_december.png`: generated by the unchanged employer
  scorer, which successfully validated both written CSVs.

Every prediction is finite and positive. CSV rates are written to six decimal
places. All raw input hashes were unchanged. The manifest records their hashes,
output hashes, the scorer hash, and model hashes/configurations.

## Saved models and reproduction

Native CatBoost models and their checked bundle manifest are stored in ignored
`artifacts/models/`. During refitting, predictions from the reloaded models were
compared against the in-memory models on all 12,000 final rows and all 31 chart
rows with absolute tolerance 1e-10. The saved artifacts can be regenerated:

```text
python -m freight.submission
```

To regenerate predictions using existing matching models without retraining:

```text
python -m freight.submission --predict-only
```

The command checks the evaluated input/code/policy and model-file checksums,
joins predictions by ID, and runs the supplied scorer CLI. Refit model binaries
may have different internal metadata/hashes; prediction values are the
reproducibility target. Preserve the model bundle manifest with its own models.

## Limits and remaining deliverables

The scorer does not calculate accuracy. The internal September-October metrics
describe models fitted only through August, not these full-data refits. The
December curve is a conditional model prediction, not a calibrated time-series
forecast or evidence that holiday effects were learned.

The formal PDF/DOCX report and 2-3 minute camera-on Loom recording remain to be
prepared. Submission files should accompany the accessible GitHub repository.
"""
    (report_dir / "submission.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"Submission report: {report_dir / 'submission.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--policy-path", type=Path, default=ROOT / "model_policy.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.policy_path.resolve(), args.model_dir.resolve(), args.output_dir.resolve(), args.predict_only)


if __name__ == "__main__":
    main()
