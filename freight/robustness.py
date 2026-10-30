"""Geographic stress tests and residual analysis on development windows only."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight.benchmark import metrics
from freight.data import TARGET, read_loads
from freight.explore import markdown_table
from freight.models import MODEL_SPECS, RateModel
from freight.splits import BACKTESTS, choose_heldout_cities, purge_cities, temporal_split


ROOT = Path(__file__).resolve().parents[1]
STRESS_MODELS = [
    "equipment_rate_per_mile", "catboost_core_mae", "catboost_market_mae",
    "catboost_full_mae", "catboost_chart_mae",
]


def validate_development_predictions(oof: pd.DataFrame, development: pd.DataFrame) -> None:
    """Reject stale labels, duplicate predictions, and final-check contamination."""
    required = ["fold", "model", "load_id", "actual_rate", "predicted_rate"]
    if list(oof.columns) != required or oof.empty:
        raise ValueError("Unexpected development prediction schema or empty predictions")
    if oof[required].isna().any().any():
        raise ValueError("Missing development prediction values")
    if oof.duplicated(["fold", "model", "load_id"]).any():
        raise ValueError("Duplicate development predictions")
    if not np.isfinite(oof[["actual_rate", "predicted_rate"]].to_numpy()).all():
        raise ValueError("Nonfinite development predictions")
    if oof.predicted_rate.le(0).any():
        raise ValueError("Nonpositive development predictions")
    expected_models = {spec.name for spec in MODEL_SPECS}
    if set(oof.model) != expected_models or set(oof.fold) != {fold[0] for fold in BACKTESTS[:2]}:
        raise ValueError("Development model or fold coverage does not match the benchmark")
    labels = development.set_index("load_id")[TARGET]
    if not oof.load_id.isin(labels.index).all():
        raise ValueError("Predictions include IDs outside the allowed development window")
    if not np.allclose(oof.actual_rate, oof.load_id.map(labels), rtol=0, atol=1e-8):
        raise ValueError("Prediction labels differ from development source data")
    for fold, train_end, validation_end in BACKTESTS[:2]:
        _, expected = temporal_split(development, train_end, validation_end)
        for model in expected_models:
            group = oof.loc[oof.fold.eq(fold) & oof.model.eq(model)]
            if set(group.load_id) != set(expected.load_id):
                raise ValueError(f"Incorrect validation IDs for {fold}/{model}")


def error_summary(group: pd.DataFrame) -> dict:
    errors = group.predicted_rate.to_numpy() - group.actual_rate.to_numpy()
    absolute = np.abs(errors)
    squared = errors ** 2
    count = max(1, int(np.ceil(len(group) * .01)))
    return {
        **metrics(group.actual_rate, group.predicted_rate),
        "bias": float(errors.mean()), "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, .9)),
        "p99_absolute_error": float(np.quantile(absolute, .99)),
        "largest_1pct_absolute_error_share": float(np.sort(absolute)[-count:].sum() / absolute.sum()) if absolute.sum() else 0.,
        "largest_1pct_squared_error_share": float(np.sort(squared)[-count:].sum() / squared.sum()) if squared.sum() else 0.,
    }


def residual_analysis(oof, development, output):
    summary = pd.DataFrame([
        {"model": name, **error_summary(group)} for name, group in oof.groupby("model")
    ]).sort_values("mae")
    detail = oof.merge(development[["load_id", "distance", "pickup", "delivery", "equipment", "weight", "date", "market_index", "quote_signal"]], on="load_id", how="left", validate="many_to_one")
    detail["actual_rate_per_mile"] = detail.actual_rate / detail.distance
    detail["absolute_error"] = (detail.predicted_rate - detail.actual_rate).abs()
    detail["rpm_band"] = "within_training_1st_99th_percentiles"
    thresholds = {}
    for fold, train_end, validation_end in BACKTESTS[:2]:
        train, _ = temporal_split(development, train_end, validation_end)
        rpm = train[TARGET] / train.distance
        low, high = rpm.quantile([.01, .99]).tolist()
        thresholds[fold] = {"training_rpm_p01": low, "training_rpm_p99": high}
        in_fold = detail.fold.eq(fold)
        detail.loc[in_fold & detail.actual_rate_per_mile.lt(low), "rpm_band"] = "below_training_1st_percentile"
        detail.loc[in_fold & detail.actual_rate_per_mile.gt(high), "rpm_band"] = "above_training_99th_percentile"
    bands = pd.DataFrame([
        {"model": name, "rpm_band": band, **error_summary(group)}
        for (name, band), group in detail.groupby(["model", "rpm_band"])
    ])
    best = summary.iloc[0].model
    worst = detail.loc[detail.model.eq(best)].nlargest(25, "absolute_error")
    summary.to_csv(output / "residual_summary.csv", index=False)
    bands.to_csv(output / "rate_per_mile_bands.csv", index=False)
    worst.to_csv(output / "largest_development_errors.csv", index=False)
    return summary, bands, thresholds


def plot_diagnostics(summary, location, output):
    color = "#075563"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    selected = summary.loc[summary.model.isin(STRESS_MODELS)].set_index("model").loc[STRESS_MODELS]
    labels = [name.replace("catboost_", "").replace("_rate_per_mile", " baseline").replace("_mae", "") for name in STRESS_MODELS]
    axes[0].barh(labels, selected.mae, color=color)
    axes[0].set(title="Original temporal backtests", xlabel="Pooled MAE ($)")
    for segment, offset, shade in [("seen", -.18, color), ("unseen", .18, "#C86826")]:
        values = location.loc[location.segment.eq(segment)].set_index("model").loc[STRESS_MODELS, "mae"]
        axes[1].barh(np.arange(len(labels)) + offset, values, height=.36, label=segment, color=shade)
    axes[1].set_yticks(np.arange(len(labels)), labels)
    axes[1].set(title="Retrained with eight cities removed", xlabel="Pooled MAE ($)")
    axes[1].legend(title="City availability", frameon=False)
    for axis in axes:
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "robustness_comparison.png", dpi=160)
    plt.close(fig)


def run(data_path, benchmark_dir, prediction_path, output, iterations, threads):
    if iterations < 1 or threads < 1:
        raise ValueError("iterations and threads must be positive")
    if not prediction_path.is_file():
        raise FileNotFoundError("Run python -m freight.benchmark first to generate development predictions")
    manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if manifest["input_sha256"] != digest or manifest["final_check_evaluated"]:
        raise ValueError("Benchmark manifest does not match the current development experiment")
    if any(config["iterations"] != iterations for config in manifest["models"].values()):
        raise ValueError("Use the same iteration count as the source development benchmark")
    all_development = read_loads(data_path, labeled=True)
    development = all_development.loc[all_development.date.le("2025-08-31")].copy()
    oof = pd.read_csv(prediction_path)
    validate_development_predictions(oof, development)
    # Ensure the supplied row-level predictions agree with committed aggregates.
    original_scores = pd.read_csv(benchmark_dir / "leaderboard.csv").set_index("model")
    for name, group in oof.groupby("model"):
        observed = metrics(group.actual_rate, group.predicted_rate)
        if not np.allclose([observed["mae"], observed["rmse"]], original_scores.loc[name, ["mae", "rmse"]].to_numpy(dtype=float)):
            raise ValueError("Development predictions do not match the source benchmark scores")
    output.mkdir(parents=True, exist_ok=True)
    first_train, _ = temporal_split(development, BACKTESTS[0][1], BACKTESTS[0][2])
    heldout = choose_heldout_cities(first_train)
    scores, predictions, fold_counts, configs = [], [], [], {}
    for fold, train_end, validation_end in BACKTESTS[:2]:
        original_train, validation = temporal_split(development, train_end, validation_end)
        train = purge_cities(original_train, heldout)
        unknown = ~validation.pickup.isin(train.pickup) | ~validation.delivery.isin(train.delivery)
        if not unknown.any() or unknown.all():
            raise ValueError("Location stress test requires seen and unseen validation loads")
        fold_counts.append({"fold": fold, "original_train_rows": len(original_train), "purged_train_rows": len(train), "validation_rows": len(validation), "unseen_city_rows": int(unknown.sum())})
        print(f"{fold}: train after city purge={len(train):,}; unseen validation loads={unknown.sum():,}", flush=True)
        for spec in MODEL_SPECS:
            if spec.name not in STRESS_MODELS:
                continue
            model = RateModel(spec, iterations=iterations, threads=threads).fit(train)
            prediction = model.predict(validation)
            configs[spec.name] = model.configuration()
            for segment, mask in [("all", np.ones(len(validation), dtype=bool)), ("seen", ~unknown.to_numpy()), ("unseen", unknown.to_numpy())]:
                scores.append({"fold": fold, "model": spec.name, "segment": segment, **metrics(validation[TARGET].to_numpy()[mask], prediction[mask])})
            predictions.append(pd.DataFrame({"fold": fold, "model": spec.name, "load_id": validation.load_id.to_numpy(), "actual_rate": validation[TARGET].to_numpy(), "predicted_rate": prediction, "segment": np.where(unknown, "unseen", "seen")}))
            print(f"  {spec.name}: unseen-city MAE=${scores[-1]['mae']:.2f}", flush=True)
    location_predictions = pd.concat(predictions, ignore_index=True)
    location = pd.DataFrame([
        {"model": name, "segment": segment, **metrics(group.actual_rate, group.predicted_rate)}
        for (name, segment), group in location_predictions.groupby(["model", "segment"])
    ])
    pd.DataFrame(scores).to_csv(output / "location_fold_metrics.csv", index=False)
    location.to_csv(output / "location_pooled_metrics.csv", index=False)
    diagnostics = ROOT / "artifacts" / "robustness"
    diagnostics.mkdir(parents=True, exist_ok=True)
    location_predictions.to_csv(diagnostics / "city_holdout_predictions.csv", index=False)
    summary, bands, thresholds = residual_analysis(oof, development, output)
    plot_diagnostics(summary, location, output)
    metadata = {
        "input_sha256": digest, "source_prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "source_benchmark_manifest_sha256": hashlib.sha256((benchmark_dir / "manifest.json").read_bytes()).hexdigest(),
        "heldout_cities": heldout, "city_selection": "eight smallest SHA256(42:city) values among cities in Jan-Apr training; target-independent",
        "fold_counts": fold_counts, "models": configs, "diagnostic_training_rpm_thresholds": thresholds,
        "final_check_evaluated": False,
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    best = summary.iloc[0]
    unseen_best = location.loc[location.segment.eq("unseen")].sort_values("mae").iloc[0]
    text = f"""# Geographic robustness and development residuals

Run `python -m freight.robustness` after `python -m freight.benchmark`.
September-October remains excluded from fitting, scoring, and model selection.

## Target-independent city holdout

Eight cities were selected by stable seeded hashes using only the cities present
in January-April training, without inspecting their labels or validation errors:
**{', '.join(heldout)}**. All loads involving any selected city, at either pickup
or delivery, were removed from each training fold. May-June and July-August
validation loads were left intact. The models were retrained from scratch.

{markdown_table(pd.DataFrame(fold_counts))}

This simulates a city being unavailable in training, including both route
directions. It also reduces the training size and changes the city mix, so it is
a stress test rather than an unbiased estimate of error on Spotter's eight new
cities. It is one fixed city subset, not a confidence interval. Seen/unseen
labels below refer to availability in the purged training fold.

![Temporal and geographic comparisons](robustness_comparison.png)

{markdown_table(location.sort_values(['segment', 'mae']))}

The lowest pooled unseen-city MAE in this stress test is
**${unseen_best.mae:,.2f}** from **{unseen_best.model}**. Consult
`location_fold_metrics.csv` for each period rather than selecting only from a
pooled score. Do not combine the stress scores with temporal scores as though
they were predictions from the same training population.

## Original temporal residuals

These diagnostics reuse the original benchmark's out-of-time predictions, not
the city-purged models. Prediction IDs, labels, source hashes, and aggregate scores
are checked before analysis. No final-input labels are used.

{markdown_table(summary[['model', 'mae', 'rmse', 'bias', 'median_absolute_error', 'p90_absolute_error', 'largest_1pct_squared_error_share']])}

For the best temporal-MAE candidate (**{best.model}**), median absolute error is
**${best.median_absolute_error:,.2f}**, and the largest 1% of errors account for
**{100 * best.largest_1pct_squared_error_share:.1f}%** of total squared error.
The full RMSE remains **${best.rmse:,.2f}**; none of these loads were deleted or
excluded from headline scores. The 25 largest errors are listed in
`largest_development_errors.csv` for inspection. Their size alone does not
establish that their labels are wrong.

`rate_per_mile_bands.csv` separates loads below, within, and above the training
fold's 1st/99th percentile rates per mile. Thresholds are fitted to training
labels only, then used for labeled retrospective diagnostics. These bands must
never be used as inference features or a reason to omit difficult validation
loads. All eight original candidate models are included in the residual audit.

## Model decision status

The temporal MAE leader remains the full-feature CatBoost candidate. MAE and
RMSE rank the market-only and full-signal variants differently, and signal
relationships vary by period. Geographic stress results add a separate check;
they do not resolve the unknown future signal regime. The chart-only candidate
has a validated path requiring exactly the supplied December feature availability.

Next, lock the prediction policy and hyperparameters using development evidence,
evaluate the reserved September-October window once, then fit on all labeled
loads and generate the final files. This checkpoint does not train a submission
model or create submission predictions.
"""
    (output / "robustness.md").write_text(text, encoding="utf-8")
    print("\nUnseen-city scores:\n" + location.loc[location.segment.eq("unseen")].sort_values("mae").to_string(index=False), flush=True)
    print(f"Report: {output / 'robustness.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=ROOT / "train-test.csv")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "reports" / "benchmark")
    parser.add_argument("--prediction-path", type=Path, default=ROOT / "artifacts" / "benchmark" / "development_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "robustness")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.data_path.resolve(), args.benchmark_dir.resolve(), args.prediction_path.resolve(), args.output_dir.resolve(), args.iterations, args.threads)


if __name__ == "__main__":
    main()
