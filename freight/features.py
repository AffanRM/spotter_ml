"""Deterministic feature construction; no labels or fitted statistics."""

import numpy as np
import pandas as pd

from freight.data import FEATURE_COLUMNS, clean_features


FEATURE_SETS = ("core", "market", "full", "chart")
CATEGORICAL = ["pickup", "delivery", "equipment"]


def build_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Support full loads and reduced December inputs using the same code path.

    Core excludes both undocumented signals. Chart additionally excludes all
    coordinates, so it needs only fields supplied in the December input file.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    required = ["pickup", "delivery", "equipment", "distance", "weight", "date"]
    if feature_set != "chart":
        required += ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    if feature_set in ("market", "full"):
        required += ["market_index"]
    if feature_set == "full":
        required += ["quote_signal"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required {feature_set} columns: {sorted(missing)}")
    work = clean_features(frame.reindex(columns=FEATURE_COLUMNS))
    dates = pd.to_datetime(work.pop("date"), errors="raise")
    if dates.isna().any():
        raise ValueError("Features require a valid date for every load")
    work["month"] = dates.dt.month
    work["day_of_week"] = dates.dt.dayofweek
    work["day_of_month"] = dates.dt.day
    work["day_of_year"] = dates.dt.dayofyear
    work["weekend"] = dates.dt.dayofweek.ge(5).astype("int8")
    if feature_set != "full":
        work = work.drop(columns="quote_signal")
    if feature_set in ("core", "chart"):
        work = work.drop(columns=["market_index", "market_index_missing"])
    if feature_set == "chart":
        work = work.drop(columns=["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    else:
        work["latitude_delta"] = work.delivery_lat - work.pickup_lat
        work["longitude_delta"] = work.delivery_lon - work.pickup_lon
        # Supplied coordinates are used as given; no external geographic facts.
        lat1, lat2 = np.radians(work.pickup_lat), np.radians(work.delivery_lat)
        delta_lon = np.radians(work.longitude_delta)
        hav = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
        work["coordinate_distance_miles"] = 3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    for column in CATEGORICAL:
        work[column] = work[column].fillna("__MISSING__").astype(str)
    return work
