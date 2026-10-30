"""Compare candidates on development backtests only; final check stays reserved."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from freight.data import TARGET, read_loads
from freight.explore import markdown_table
from freight.models import MODEL_SPECS, RateModel
from freight.splits import BACKTESTS, temporal_split


ROOT = Path(__file__).resolve().parents[1]


def metrics(actual, predicted):
    return {
        "rows": len(actual), "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def segments(train, validation):
    unseen = ~validation.pickup.isin(train.pickup) | ~validation.delivery.isin(train.delivery)
    return {
        "month": validation.date.dt.strftime("%Y-%m"),
        "equipment": validation.equipment,
        "distance": pd.cut(validation.distance, [0, 500, 1000, 2000, np.inf], labels=["0-500", "500-1000", "1000-2000", "2000+"]),
        "weight_quality": pd.Series(np.where(validation.weight.isna(), "missing", np.where(validation.weight.le(0), "invalid", "valid")), index=validation.index),
        "market_quality": validation.market_index.isna().map({True: "missing", False: "present"}),
        "city_novelty": unseen.map({True: "unseen", False: "seen"}),
    }


def run(data_path: Path, output: Path, iterations: int, threads: int):
    if iterations < 1 or threads < 1:
        raise ValueError("iterations and threads must be positive")
    output.mkdir(parents=True, exist_ok=True)
    all_development = read_loads(data_path, labeled=True)
    # The final-check labels are never passed to a candidate or metric function.
    development = all_development.loc[all_development.date.le("2025-08-31")].copy()
    scores, breakdowns, predictions, configurations = [], [], [], {}
    for fold, train_end, validation_end in BACKTESTS[:2]:
        train, validation = temporal_split(development, train_end, validation_end)
        print(f"{fold}: train={len(train):,}; validate={len(validation):,}", flush=True)
        for spec in MODEL_SPECS:
            start = perf_counter()
            model = RateModel(spec, iterations=iterations, threads=threads).fit(train)
            fitted = perf_counter()
            prediction = model.predict(validation)
            elapsed = perf_counter() - start
            score = {"fold": fold, "model": spec.name, **metrics(validation[TARGET], prediction), "fit_seconds": fitted - start, "total_seconds": elapsed}
            scores.append(score)
            configurations[spec.name] = model.configuration()
            predictions.append(pd.DataFrame({"fold": fold, "model": spec.name, "load_id": validation.load_id.to_numpy(), "actual_rate": validation[TARGET].to_numpy(), "predicted_rate": prediction}))
            for dimension, labels in segments(train, validation).items():
                for label in sorted(labels.dropna().unique()):
                    mask = labels.eq(label).to_numpy()
                    breakdowns.append({"fold": fold, "model": spec.name, "dimension": dimension, "segment": str(label), **metrics(validation[TARGET].to_numpy()[mask], prediction[mask])})
            print(f"  {spec.name}: MAE=${score['mae']:.2f}; RMSE=${score['rmse']:.2f}; {elapsed:.1f}s", flush=True)
    fold_scores = pd.DataFrame(scores)
    oof = pd.concat(predictions, ignore_index=True)
    leaderboard = pd.DataFrame([
        {"model": name, **metrics(group.actual_rate, group.predicted_rate)}
        for name, group in oof.groupby("model")
    ]).sort_values(["mae", "rmse"]).reset_index(drop=True)
    fold_scores.to_csv(output / "fold_metrics.csv", index=False)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    pd.DataFrame(breakdowns).to_csv(output / "segment_metrics.csv", index=False)
    # Large row-level diagnostics are reproducible and kept out of Git.
    diagnostics = ROOT / "artifacts" / "benchmark"
    diagnostics.mkdir(parents=True, exist_ok=True)
    oof.to_csv(diagnostics / "development_predictions.csv", index=False)
    manifest = {
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "catboost"]},
        "models": configurations, "folds": [list(fold) for fold in BACKTESTS[:2]],
        "selection_rule": "pooled development MAE ascending, then RMSE ascending",
        "final_check_evaluated": False,
        "best_development_model": leaderboard.iloc[0].model,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    baseline_mae = float(leaderboard.set_index("model").loc["equipment_rate_per_mile", "mae"])
    best = leaderboard.iloc[0]
    improvement = 100 * (1 - best.mae / baseline_mae)
    text = f"""# Development benchmark

Run: `python -m freight.benchmark --iterations {iterations} --threads {threads}`.
All candidates use fixed parameters and random seed 42. No final-input labels
exist, and September-October model performance has not been evaluated.

## Pooled development results

{markdown_table(leaderboard)}

Ranked by MAE in dollars, with RMSE as a secondary comparison because the hidden
metric is unspecified. Scores pool the nonoverlapping May-June and July-August
validation rows; each load has one out-of-time prediction per model. The best
development MAE is **${best.mae:,.2f}** from **{best.model}**, a **{improvement:.1f}%**
reduction from the equipment rate-per-mile baseline. This is a development
comparison, not an unbiased final performance estimate.

## Individual temporal backtests

{markdown_table(fold_scores.drop(columns=['fit_seconds', 'total_seconds']))}

Training expands January-April -> May-June and January-June -> July-August.
Model selection uses these two folds only. Imputation, scaling, one-hot encoding,
equipment medians, and all estimator fits use the respective training rows.
There is no early stopping against validation labels and no outlier deletion.

## Candidate definitions

- `median_rate`: one median training rate for every load.
- `equipment_rate_per_mile`: median training rate/mile by equipment, multiplied
  by distance; unseen equipment falls back to the global training median rate/mile.
- `ridge_core_rate_per_mile`: linear regularized rate/mile regression with
  training-fitted median imputation, numeric scaling, and unknown-safe one-hot encoding.
- `catboost_core_mae`: raw-dollar MAE objective; supplied coordinates, cities,
  equipment, distance, cleaned weight/flags, and calendar features; neither signal.
- `catboost_market_mae`: core plus market index and its missingness indicator.
- `catboost_full_mae`: market features plus quote signal.
- `catboost_core_rmse`: same core features, squared-error training objective.
- `catboost_chart_mae`: only December-available features and derived calendar/
  weight flags. No coordinates, market index, or quote signal are required.

CatBoost uses {iterations} iterations, depth 6, learning rate 0.05, and L2 leaf
regularization 5. All predictions have a fixed $0.01 lower bound to satisfy
positive-rate contracts. No model artifacts are serialized at this stage.

## Diagnostics and limitations

`segment_metrics.csv` reports month, equipment, distance band, weight quality,
market missingness, and city novelty for each model/fold. These time folds have
no unseen cities; a separate labeled location holdout is still required to
assess that risk. The raw out-of-fold predictions are reproducible in ignored
`artifacts/benchmark/development_predictions.csv`.

The quote-signal ablation measures sensitivity to its changing relationship.
The chart candidate is trained and evaluated without all six omitted inputs,
so its scores assess the reduced-input path directly. Ranking one overall model
does not settle which model to use for December chart rows.

Before finalizing: inspect fold consistency and segment errors; run a location
holdout; lock the model decisions; evaluate September-October once; then refit
on all labeled development data and generate submission predictions.
"""
    (output / "benchmark.md").write_text(text, encoding="utf-8")
    print("\n" + leaderboard.to_string(index=False), flush=True)
    print(f"Report: {output / 'benchmark.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "train-test.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "benchmark")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.data_path.resolve(), args.output_dir.resolve(), args.iterations, args.threads)


if __name__ == "__main__":
    main()
