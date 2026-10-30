"""Expanding-window backtests that imitate the two-month prediction horizon."""

import pandas as pd


BACKTESTS = [
    ("development_1", "2025-04-30", "2025-06-30"),
    ("development_2", "2025-06-30", "2025-08-31"),
    ("final_check", "2025-08-31", "2025-10-31"),
]


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
