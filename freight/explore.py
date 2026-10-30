"""Run with: python -m freight.explore (no input files are modified)."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight.data import FEATURE_COLUMNS, TARGET, clean_features, read_loads
from freight.splits import BACKTESTS, temporal_split


ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    "train-test.csv", "validation.csv", "validation-predictions-template.csv",
    "december-chart-inputs.csv",
]


def markdown_table(frame: pd.DataFrame, *, index: bool = False) -> str:
    table = frame.reset_index() if index else frame.copy()
    def display(value):
        if pd.isna(value):
            return "-"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, table.columns)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(table.columns)) + " |")
    lines.extend("| " + " | ".join(map(display, row)) + " |" for row in table.itertuples(index=False, name=None))
    return "\n".join(lines)


def quality_summary(frame: pd.DataFrame) -> dict:
    coordinates = pd.concat([
        frame[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        frame[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ])
    inconsistent = coordinates.groupby("city")[["lat", "lon"]].nunique().gt(1).any(axis=1)
    return {
        "rows": len(frame),
        "date_min": str(frame.date.min().date()),
        "date_max": str(frame.date.max().date()),
        "missing_by_column": {c: int(n) for c, n in frame.isna().sum().items() if n},
        "nonpositive_weights": int(frame.weight.le(0).sum()),
        "duplicate_ids": int(frame.load_id.duplicated().sum()),
        "duplicate_feature_rows": int(frame[FEATURE_COLUMNS].duplicated().sum()),
        "pickup_cities": int(frame.pickup.nunique()),
        "delivery_cities": int(frame.delivery.nunique()),
        "cities_with_inconsistent_coordinates": int(inconsistent.sum()),
    }


def novelty(reference: pd.DataFrame, future: pd.DataFrame) -> dict:
    result = {}
    for column in ["pickup", "delivery", "equipment"]:
        unseen = ~future[column].isin(reference[column])
        result[f"unseen_{column}_values"] = sorted(future.loc[unseen, column].unique().tolist())
        result[f"unseen_{column}_rows"] = int(unseen.sum())
    known_lanes = pd.MultiIndex.from_frame(reference[["pickup", "delivery"]])
    lanes = pd.MultiIndex.from_frame(future[["pickup", "delivery"]])
    result["unseen_directed_lane_rows"] = int((~lanes.isin(known_lanes)).sum())
    result["either_city_unseen_rows"] = int((
        ~future.pickup.isin(reference.pickup) | ~future.delivery.isin(reference.delivery)
    ).sum())
    return result


def monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.assign(month=frame.date.dt.strftime("%Y-%m"))
    if TARGET in frame:
        work["rate_per_mile"] = work[TARGET] / work.distance
    records = []
    for month, group in work.groupby("month", sort=True):
        record = {
            "month": month, "rows": len(group),
            "missing_weight_pct": 100 * group.weight.isna().mean(),
            "invalid_weight_pct": 100 * group.weight.le(0).mean(),
            "missing_market_pct": 100 * group.market_index.isna().mean(),
            "median_distance": group.distance.median(),
            "median_market": group.market_index.median(),
            "median_quote": group.quote_signal.median(),
        }
        if TARGET in frame:
            record.update({
                "median_rate": group[TARGET].median(),
                "median_rate_per_mile": group.rate_per_mile.median(),
                "quote_rpm_correlation": group.quote_signal.corr(group.rate_per_mile),
                "market_rpm_correlation": group.market_index.corr(group.rate_per_mile),
            })
        records.append(record)
    return pd.DataFrame(records)


def save_charts(train, final, monthly, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    teal, orange = "#075563", "#C86826"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    counts = pd.concat([train, final]).groupby(pd.concat([train, final]).date.dt.to_period("M")).size()
    colors = [teal if month.month <= 10 else orange for month in counts.index]
    axes[0, 0].bar(counts.index.astype(str), counts.values, color=colors)
    axes[0, 0].tick_params(axis="x", rotation=55)
    axes[0, 0].set(title="Load counts: development (teal), final (orange)", ylabel="Loads")
    axes[0, 1].hist(train[TARGET], bins=70, color=teal)
    axes[0, 1].set(title="Development target: full range, including extremes", xlabel="Posted rate ($)", ylabel="Loads")
    axes[0, 1].set_yscale("log")
    sample = train.sample(min(6000, len(train)), random_state=42)
    axes[1, 0].scatter(sample.distance, sample[TARGET], s=5, alpha=.25, color=teal)
    axes[1, 0].set(title="Rate versus distance (seeded sample)", xlabel="Distance (miles)", ylabel="Posted rate ($)")
    for frame, label, color in [(train, "Development", teal), (final, "Final inputs", orange)]:
        values = [100 * frame.weight.isna().mean(), 100 * frame.weight.le(0).mean(), 100 * frame.market_index.isna().mean()]
        offset = -.18 if label == "Development" else .18
        axes[1, 1].bar(np.arange(3) + offset, values, width=.36, label=label, color=color)
    axes[1, 1].set_xticks(range(3), ["Missing weight", "Invalid weight", "Missing market"])
    axes[1, 1].set(title="Input quality differs across datasets", ylabel="Rows (%)")
    axes[1, 1].legend(frameon=False)
    fig.savefig(output / "data_overview.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    x = np.arange(len(monthly))
    axes[0].plot(x, monthly.median_rate_per_mile, "o-", label="Actual rate / mile", color=teal)
    axes[0].plot(x, monthly.median_quote, "s--", label="Quote signal (units undocumented)", color=orange)
    axes[0].set(title="Monthly medians: numerical comparison, not proof of equivalent units", ylabel="Median value")
    axes[0].legend(frameon=False)
    axes[1].bar(x, monthly.quote_rpm_correlation, color=[teal if v >= 0 else orange for v in monthly.quote_rpm_correlation])
    axes[1].axhline(0, color="#666666", linewidth=.8)
    axes[1].set(title="Quote signal versus rate per mile: direction changes by month", ylabel="Pearson correlation", ylim=(-1, 1))
    for axis in axes:
        axis.set_xticks(x, monthly.month, rotation=35)
    fig.savefig(output / "signal_stability.png", dpi=160)
    plt.close(fig)


def run(data_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    train = read_loads(data_dir / "train-test.csv", labeled=True)
    final = read_loads(data_dir / "validation.csv", labeled=False)
    template = pd.read_csv(data_dir / "validation-predictions-template.csv")
    december = pd.read_csv(data_dir / "december-chart-inputs.csv")
    if list(template.columns) != ["load_id", "predicted_rate"] or template.load_id.duplicated().any():
        raise ValueError("Invalid prediction template schema or duplicate IDs")
    if len(template) != len(final) or set(template.load_id) != set(final.load_id):
        raise ValueError("Template and final IDs must match exactly")
    if set(train.load_id) & set(final.load_id):
        raise ValueError("Development and final IDs overlap")
    if train.date.max() >= final.date.min():
        raise ValueError("Expected all final loads to be later than development loads")
    expected_december_columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if list(december.columns) != expected_december_columns:
        raise ValueError("Invalid December input schema")
    december_dates = pd.to_datetime(december.date, errors="raise")
    if len(december) != 31 or set(december_dates) != set(pd.date_range("2025-12-01", "2025-12-31")):
        raise ValueError("December inputs must have exactly one load per December day")
    for column, value in {"pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360, "equipment": "Dry Van", "weight": 32000}.items():
        if not december[column].eq(value).all():
            raise ValueError(f"Unexpected fixed December {column}")

    monthly = monthly_summary(train)
    monthly_final = monthly_summary(final)
    splits = []
    split_novelty = {}
    for name, train_end, validation_end in BACKTESTS:
        past, future = temporal_split(train, train_end, validation_end)
        splits.append({"name": name, "train_end": train_end, "validation_start": str(future.date.min().date()), "validation_end": validation_end, "train_rows": len(past), "validation_rows": len(future)})
        split_novelty[name] = novelty(past, future)
    cleaned = clean_features(train)
    target_summary = train[TARGET].describe(percentiles=[.01, .05, .5, .95, .99, .999]).to_dict()
    rate_per_mile = train[TARGET] / train.distance
    quantiles = train[TARGET].quantile([.25, .75])
    iqr_upper = quantiles.loc[.75] + 1.5 * (quantiles.loc[.75] - quantiles.loc[.25])
    # Diagnostic only. This full-data threshold must never become a learned
    # training filter; any learned thresholds belong inside each training fold.
    report = {
        "python_version": platform.python_version(),
        "input_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in INPUTS},
        "development": quality_summary(train), "final_inputs": quality_summary(final),
        "final_novelty": novelty(train, final), "backtest_novelty": split_novelty,
        "target_summary": target_summary,
        "target_iqr_upper_diagnostic": float(iqr_upper),
        "rows_above_target_iqr_upper": int(train[TARGET].gt(iqr_upper).sum()),
        "rate_per_mile_summary": rate_per_mile.describe(percentiles=[.01, .5, .95, .99]).to_dict(),
        "target_distance_correlation": float(train[TARGET].corr(train.distance)),
        "quote_rpm_overall_correlation": float(train.quote_signal.corr(rate_per_mile)),
        "missing_weights_after_cleaning": int(cleaned.weight.isna().sum()),
        "backtests": splits,
        "monthly_development": monthly.to_dict(orient="records"),
        "monthly_final_inputs": monthly_final.to_dict(orient="records"),
        "december_missing_model_features": [c for c in FEATURE_COLUMNS if c not in december],
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_charts(train, final, monthly, output)
    quality = pd.DataFrame([
        {"dataset": name, "rows": len(frame), "missing_weight": int(frame.weight.isna().sum()), "nonpositive_weight": int(frame.weight.le(0).sum()), "missing_market": int(frame.market_index.isna().sum())}
        for name, frame in [("Development", train), ("Final inputs", final)]
    ])
    equipment = train.assign(rate_per_mile=rate_per_mile).groupby("equipment").agg(rows=(TARGET, "size"), median_rate=(TARGET, "median"), median_rate_per_mile=("rate_per_mile", "median"))
    text = f"""# Exploratory analysis and data-quality audit

Generated by `python -m freight.explore`. Inputs are read-only. Exact input
SHA-256 hashes and machine-readable results are in `audit.json`.

## Data and prediction horizon

Development: **{len(train):,}** loads, {train.date.min().date()} through {train.date.max().date()}.
Final inputs: **{len(final):,}** loads, {final.date.min().date()} through {final.date.max().date()}.
The final inputs have no labels. Their feature distributions are inspected only
to check inference compatibility; they cannot provide accuracy measurements.
All template IDs match; development and final IDs are disjoint.

![Data overview](data_overview.png)

## Quality and cleaning policy

{markdown_table(quality)}

Both main datasets have unique IDs and no duplicate feature rows. Coordinates
are within valid latitude/longitude bounds and consistent per city within each
dataset; this does not prove their real-world geographic accuracy. No external
geocoding or coordinate replacement has been performed.

Treat nonpositive weights as unknown, with separate missing/invalid indicators.
Do not turn negative weights positive without evidence. The stateless cleaner
leaves missing values for a later model or a fold-fitted imputer. It removes
`load_id` and `posted_rate` from the feature frame. Market-index missingness also
gets an indicator. Do not drop final prediction rows.

Development weights missing after this cleaning: **{cleaned.weight.isna().sum():,}**.

## Target and equipment

Median rate: **${train[TARGET].median():,.2f}**; observed range:
**${train[TARGET].min():,.2f}-${train[TARGET].max():,.2f}**.
Pearson correlation of distance with posted rate: **{report['target_distance_correlation']:.4f}**.
There are **{report['rows_above_target_iqr_upper']:,}** rates above the exploratory
upper IQR fence (${iqr_upper:,.2f}). These are candidates for investigation, not
confirmed errors. Retain labels initially; compare robust objectives and inspect
residuals rather than deleting expensive loads. Any learned clipping or outlier
threshold must use the training fold only, never validation labels.

{markdown_table(equipment, index=True)}

## Signal stability and leakage uncertainty

![Monthly signal stability](signal_stability.png)

{markdown_table(monthly[['month', 'rows', 'median_rate_per_mile', 'median_quote', 'quote_rpm_correlation', 'median_market']])}

Quote-signal correlation with rate per mile changes sign across months. Overall
correlation is only **{report['quote_rpm_overall_correlation']:.4f}** and masks these
changes. This is evidence of an unstable observed relationship, not proof of
target leakage. Signal generation and availability at quote time are undocumented.
Compare models with and without quote signal and market index on chronological
backtests. Do not create month-specific sign corrections using holdout labels.
No yearly seasonal effect can be reliably estimated from this partial single year.

## New locations and lanes

Final pickup cities absent from training: {', '.join(report['final_novelty']['unseen_pickup_values'])}.
Final delivery cities absent from training: {', '.join(report['final_novelty']['unseen_delivery_values'])}.
Final rows with either city unseen: **{report['final_novelty']['either_city_unseen_rows']:,}**.
Final rows with an unseen directed pickup-delivery lane: **{report['final_novelty']['unseen_directed_lane_rows']:,}**.
Use features and encoders that support unfamiliar locations. In addition to time
backtests, simulate location holdouts using labeled data; temporal backtests alone
may not represent this shift. Unseen counts for each proposed temporal split are
recorded in `audit.json`.

## Proposed validation protocol

{markdown_table(pd.DataFrame(splits))}

Use expanding training windows and two-month validation horizons. Use the first
two windows for model/feature decisions. Reserve September-October for a final
performance check after those decisions; do not tune repeatedly against it.
Dataset-level exploratory summaries include all development months, but no model
performance on this final check has been inspected. Fit preprocessing, encoders,
imputers, and any target transformations within each training fold. Use MAE and
RMSE in dollars, plus R-squared, and report errors by month, equipment, distance,
missingness, and unseen locations. The employer's hidden metric is unspecified.
After selection and the final check, refit on all eligible labeled loads.

## December chart inference

The 31 fixed rows omit: {', '.join(report['december_missing_model_features'])}.
Compare a model using only chart-available features against a full-feature model
with an explicitly validated missing-feature path. A city-coordinate mapping,
if used, must be learned from training rows only. Future quote signals and market
indices cannot be recovered from labels or fabricated as observed values. Validate
the reduced-input path by masking those features on chronological holdouts.
Use the supplied scorer to generate the required final chart after predictions.

## Reproduce

```powershell
python -m freight.explore
python -m unittest discover -s tests -v
```

This checkpoint performs analysis and validates cleaning/splitting utilities.
No submission predictions or final model have been generated yet.
"""
    (output / "exploration.md").write_text(text, encoding="utf-8")
    print(f"Audited {len(train):,} labeled and {len(final):,} unlabeled loads.")
    print(f"Report: {output / 'exploration.md'}")
    print(f"Audit: {output / 'audit.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "eda")
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
