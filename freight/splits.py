"""Expanding-window backtests that imitate the two-month prediction horizon."""

import hashlib

import pandas as pd


BACKTESTS = [
    ("development_1", "2025-04-30", "2025-06-30"),
    ("development_2", "2025-06-30", "2025-08-31"),
    ("final_check", "2025-08-31", "2025-10-31"),
]


def choose_heldout_cities(frame: pd.DataFrame, *, count: int = 8, seed: int = 42) -> list[str]:
    """Choose cities by a stable seeded hash, independent of labels and row order.

    Call on the earliest training window so even the list of candidate cities
    is known before any validation date. Python's process-randomized hash is
    deliberately avoided to keep the selection reproducible across computers.
    """
    cities = set(frame.pickup.dropna()) | set(frame.delivery.dropna())
    if count < 1 or count >= len(cities):
        raise ValueError("City holdout count must be between 1 and number of cities minus 1")
    return sorted(cities, key=lambda city: hashlib.sha256(f"{seed}:{city}".encode()).hexdigest())[:count]


def purge_cities(frame: pd.DataFrame, heldout_cities: list[str]) -> pd.DataFrame:
    """Remove every training load involving a held-out city in either role."""
    if not heldout_cities:
        raise ValueError("Provide at least one held-out city")
    keep = ~frame.pickup.isin(heldout_cities) & ~frame.delivery.isin(heldout_cities)
    result = frame.loc[keep].copy()
    if result.empty:
        raise ValueError("City purge removed every training row")
    return result


def temporal_split(
    frame: pd.DataFrame, train_end: str, validation_end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return disjoint training/past and validation/future copies.

    Cutoffs are inclusive calendar dates; later records are excluded. Require
    complete timestamp values so a missing date cannot silently lose a row.
    """
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("Cannot split rows with missing dates")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Expected calendar dates without time components")
    train_cutoff, validation_cutoff = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must precede validation_end")
    train = frame.loc[dates.le(train_cutoff)].copy()
    validation = frame.loc[dates.gt(train_cutoff) & dates.le(validation_cutoff)].copy()
    if train.empty or validation.empty:
        raise ValueError("Both temporal partitions must contain rows")
    return train, validation
