"""Prevent future rows or a shared calendar day entering both partitions."""

import unittest

import pandas as pd

from freight.splits import temporal_split


class TemporalSplitTests(unittest.TestCase):
    def test_boundaries_disjointness_and_exclusion_of_later_data(self):
        frame = pd.DataFrame({
            "load_id": ["early", "cutoff_a", "cutoff_b", "future", "end", "later"],
            "date": pd.to_datetime(["2025-01-01", "2025-04-30", "2025-04-30", "2025-05-01", "2025-06-30", "2025-07-01"]),
        }).sample(frac=1, random_state=42)
        original = frame.copy(deep=True)
        train, validation = temporal_split(frame, "2025-04-30", "2025-06-30")
        self.assertEqual(set(train.load_id), {"early", "cutoff_a", "cutoff_b"})
        self.assertEqual(set(validation.load_id), {"future", "end"})
        self.assertLess(train.date.max(), validation.date.min())
        self.assertFalse(set(train.load_id) & set(validation.load_id))
        train.loc[:, "load_id"] = "changed"
        pd.testing.assert_frame_equal(frame, original)

    def test_missing_or_timed_dates_and_invalid_windows_raise(self):
        for dates, train_end, validation_end in [
            (["2025-01-01", None], "2025-01-31", "2025-02-28"),
            (["2025-01-01 12:00:00", "2025-02-01 12:00:00"], "2025-01-31", "2025-02-28"),
            (["2025-01-01", "2025-02-01"], "2025-02-28", "2025-01-31"),
            (["2025-01-01", "2025-02-01"], "2025-03-31", "2025-04-30"),
        ]:
            with self.subTest(dates=dates, train_end=train_end):
                with self.assertRaises(ValueError):
                    temporal_split(pd.DataFrame({"date": dates}), train_end, validation_end)


if __name__ == "__main__":
    unittest.main()
