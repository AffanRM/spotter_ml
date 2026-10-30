"""Input contracts and stateless cleaning shared by analysis and later modeling."""

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
    "delivery_lon", "distance", "equipment", "weight", "date",
    "market_index", "quote_signal",
]
NUMERIC_COLUMNS = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "market_index", "quote_signal",
]
TARGET = "posted_rate"


def read_loads(path: Path, *, labeled: bool) -> pd.DataFrame:
    """Reject structural errors; retain recoverable missing/invalid weights."""
    frame = pd.read_csv(path)
    expected = ["load_id", *FEATURE_COLUMNS, *([TARGET] if labeled else [])]
    if list(frame.columns) != expected:
        raise ValueError(f"{path.name}: expected columns {expected}")
    if frame.empty:
        raise ValueError(f"{path.name}: no loads found")
    if frame.load_id.isna().any() or frame.load_id.duplicated().any():
        raise ValueError(f"{path.name}: missing or duplicate load_id")
    for column in NUMERIC_COLUMNS + ([TARGET] if labeled else []):
        converted = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].notna() & converted.isna()).any():
            raise ValueError(f"{path.name}: nonnumeric {column}")
        if np.isinf(converted).any():
            raise ValueError(f"{path.name}: infinite {column}")
        frame[column] = converted
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    required = ["load_id", *FEATURE_COLUMNS]
    required = [c for c in required if c not in ["weight", "market_index"]]
    if frame[required].isna().any().any():
        raise ValueError(f"{path.name}: missing required inputs")
    for column in ["pickup", "delivery", "equipment"]:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{path.name}: blank {column}")
    for column in ["distance", "market_index", "quote_signal"]:
        if frame[column].dropna().le(0).any():
            raise ValueError(f"{path.name}: nonpositive {column}")
    for column, limit in [
        ("pickup_lat", 90), ("delivery_lat", 90),
        ("pickup_lon", 180), ("delivery_lon", 180),
    ]:
        if frame[column].abs().gt(limit).any():
            raise ValueError(f"{path.name}: out-of-range {column}")
    if labeled and (frame[TARGET].isna().any() or frame[TARGET].le(0).any()):
        raise ValueError(f"{path.name}: target must be finite and positive")
    return frame


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark nonpositive weights unknown, preserving the reason in indicators.

    No statistics are learned here. Any later imputer must fit on the training
    fold only. Do not take absolute weights: the intended value is unknown.
    IDs and labels are excluded to prevent accidental use as model features.
    """
    result = frame[FEATURE_COLUMNS].copy()
    result["weight_missing"] = result.weight.isna().astype("int8")
    result["weight_invalid"] = result.weight.le(0).astype("int8")
    result["market_index_missing"] = result.market_index.isna().astype("int8")
    result.loc[result.weight.le(0), "weight"] = np.nan
    return result
