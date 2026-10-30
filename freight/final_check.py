"""Evaluate the predeclared models on September-October without retuning."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight.benchmark import metrics, segments
from freight.data import TARGET, read_loads
from freight.explore import markdown_table
from freight.features import build_features
from freight.policy import ROOT, load_policy, make_policy_model
from freight.splits import temporal_split


IMPLEMENTATION_FILES = [
    "freight/data.py", "freight/features.py", "freight/models.py",
    "freight/policy.py", "freight/splits.py", "freight/benchmark.py",
    "freight/final_check.py",
]
CHART_INPUT_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]


def experiment_identity(data_path, policy_path):
    provenance = {
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in IMPLEMENTATION_FILES},
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "catboost"]},
    }
    provenance["experiment_id"] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    return provenance


def run(data_path, policy_path, output):
    policy = load_policy(policy_path)
    identity = experiment_identity(data_path, policy_path)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["experiment_id"] != identity["experiment_id"]:
            raise ValueError("Existing final-check results use a different policy, code, input, or environment. Preserve them; changed experiments are not a new untouched holdout.")
    # A policy file and decisions document exist before labels are evaluated.
    development = read_loads(data_path, labeled=True)
    train, validation = temporal_split(development, policy["train_end"], policy["validation_end"])
    output.mkdir(parents=True, exist_ok=True)
    print(f"Frozen experiment {identity['experiment_id'][:12]}", flush=True)
    print(f"Training: {len(train):,} rows through {train.date.max().date()}", flush=True)
    print(f"Final check: {len(validation):,} rows, {validation.date.min().date()} to {validation.date.max().date()}", flush=True)
    scores, breakdowns, predictions, configurations, importances = [], [], [], {}, []
    for role in ["reference", "main", "chart"]:
        model = make_policy_model(policy, role).fit(train)
        # Exercise the exact reduced-input contract used by the December file.
        inference = validation[CHART_INPUT_COLUMNS] if role == "chart" else validation.drop(columns=TARGET)
        prediction = model.predict(inference)
        score = {"role": role, "model": model.spec.name, **metrics(validation[TARGET], prediction)}
        scores.append(score)
        configurations[role] = model.configuration()
        predictions.append(pd.DataFrame({"role": role, "model": model.spec.name, "load_id": validation.load_id.to_numpy(), "actual_rate": validation[TARGET].to_numpy(), "predicted_rate": prediction}))
        for dimension, labels in segments(train, validation).items():
            for label in sorted(labels.dropna().unique()):
                mask = labels.eq(label).to_numpy()
                breakdowns.append({"role": role, "model": model.spec.name, "dimension": dimension, "segment": str(label), **metrics(validation[TARGET].to_numpy()[mask], prediction[mask])})
        if model.spec.kind == "catboost":
            features = build_features(train, model.spec.feature_set)
            for feature, importance in zip(features.columns, model.estimator_.get_feature_importance()):
                importances.append({"role": role, "feature": feature, "training_prediction_value_change_importance": float(importance)})
        print(f"  {role}: MAE=${score['mae']:.2f}; RMSE=${score['rmse']:.2f}; R2={score['r2']:.4f}", flush=True)
    scores = pd.DataFrame(scores)
    breakdowns = pd.DataFrame(breakdowns)
    predictions = pd.concat(predictions, ignore_index=True)
    scores.to_csv(output / "metrics.csv", index=False)
    breakdowns.to_csv(output / "segment_metrics.csv", index=False)
    pd.DataFrame(importances).to_csv(output / "feature_importance.csv", index=False)
    artifacts = ROOT / "artifacts" / "final_check"
    artifacts.mkdir(parents=True, exist_ok=True)
    prediction_path = artifacts / "predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    metadata = {
        **identity, "frozen_policy": policy, "models": configurations,
        "train_rows": len(train), "validation_rows": len(validation),
        "train_date_min": str(train.date.min().date()), "train_date_max": str(train.date.max().date()),
        "validation_date_min": str(validation.date.min().date()), "validation_date_max": str(validation.date.max().date()),
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "final_check_evaluated": True,
        "decision": "Retain predeclared configurations; no selection or retuning from final-check scores",
    }
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    main_predictions = predictions.loc[predictions.role.eq("main")]
    sample = main_predictions.sample(min(4000, len(main_predictions)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].scatter(sample.actual_rate, sample.predicted_rate, s=8, alpha=.25, color="#075563")
    limit = max(sample.actual_rate.max(), sample.predicted_rate.max())
    axes[0].plot([0, limit], [0, limit], "--", color="#C86826", linewidth=1)
    axes[0].set(title="Main model: seeded final-check sample", xlabel="Actual rate ($)", ylabel="Predicted rate ($)")
    monthly = breakdowns.loc[breakdowns.dimension.eq("month")]
    months = sorted(monthly.segment.unique())
    for offset, role, color in [(-.25, "reference", "#8B9B9F"), (0, "main", "#075563"), (.25, "chart", "#C86826")]:
        values = monthly.loc[monthly.role.eq(role)].set_index("segment").loc[months, "mae"]
        axes[1].bar(np.arange(len(months)) + offset, values, width=.25, label=role, color=color)
    axes[1].set_xticks(np.arange(len(months)), months)
    axes[1].set(title="Frozen paths by final-check month", ylabel="MAE ($)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "final_check.png", dpi=160)
    plt.close(fig)
    indexed = scores.set_index("role")
    improvement = 100 * (1 - indexed.loc["main", "mae"] / indexed.loc["reference", "mae"])
    text = f"""# Frozen September-October evaluation

The policy in `model_policy.json` and its rationale in `MODEL_DECISIONS.md` were
recorded before evaluating this window. Experiment ID:
`{identity['experiment_id']}`. Source, configuration, implementation, environment,
and prediction hashes are recorded in `manifest.json`.

## Results

{markdown_table(scores)}

Training: **{len(train):,}** loads from {train.date.min().date()} through
{train.date.max().date()}. Evaluation: **{len(validation):,}** later loads from
{validation.date.min().date()} through {validation.date.max().date()}.
The main model's MAE reduction versus the predeclared reference is
**{improvement:.1f}%**. These are out-of-time internal metrics, not Spotter's hidden
submission scores. The employer's evaluation metric is unspecified.

![Final-check diagnostics](final_check.png)

## Monthly results

{markdown_table(monthly.drop(columns='dimension'))}

## Method and interpretation

- Main: full-feature CatBoost MAE model for all loads, including unseen cities.
- Chart: separately trained CatBoost MAE model using only the six December input
  fields. All six unavailable features were removed before inference here;
  no future market/quote signals or coordinates were supplied to that model.
- Reference: training median rate per mile by equipment, with a global fallback.
- All model fits used January-August rows only, fixed parameters, and no early
  stopping against this window. Every evaluation load is included. No target
  outliers were removed and no post-evaluation adjustments were made.
- `segment_metrics.csv` includes equipment, distance, missingness, month, and
  city novelty. This temporal window has no unseen cities; geographic risk is
  assessed separately in the development city stress test.
- `feature_importance.csv` contains training-derived prediction-value-change
  importance, not holdout permutation importance or causal evidence. Correlated
  features can share or dilute these values. It is descriptive and does not
  drive a new feature selection round.

## Frozen decision and next step

Retain the predeclared main and chart models and their parameters regardless of
these results. Refit them on all 48,000 eligible labeled loads, generate the
12,000 final predictions and 31 fixed December predictions, and validate them
with the unchanged supplied scorer. Those refitted models have different
training data; the metrics above describe the January-August fits only.

Re-running `python -m freight.final_check` with the same experiment reproduces
this evaluation; it does not create another independent test set. The command
refuses to overwrite an existing report from a different experiment.
"""
    (output / "final_check.md").write_text(text, encoding="utf-8")
    print(f"Report: {output / 'final_check.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "train-test.csv")
    parser.add_argument("--policy-path", type=Path, default=ROOT / "model_policy.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "final_check")
    args = parser.parse_args()
    run(args.data_path.resolve(), args.policy_path.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
