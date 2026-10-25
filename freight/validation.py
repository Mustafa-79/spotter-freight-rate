"""Time-based validation design and accuracy measures.

Spotter's loads (November–December 2025) come after the labelled data, so every
check trains on the past and predicts the following two months:

* four walk-forward folds inside January–August drive every modelling decision;
* September–October is a test period, scored once after the model is chosen;
* the final model is retrained on all of January–October.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_START = pd.Timestamp("2025-01-01")
TEST_START = pd.Timestamp("2025-09-01")
DATA_END = pd.Timestamp("2025-10-31")
HORIZON_MONTHS = 2  # Spotter's prediction set spans two months


@dataclass(frozen=True)
class Split:
    """Train on [train_start, train_end], evaluate on [eval_start, eval_end] (dates inclusive)."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp

    def masks(self, dates: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Boolean masks selecting the training and evaluation rows."""
        dates = pd.to_datetime(dates)
        return (dates.between(self.train_start, self.train_end),
                dates.between(self.eval_start, self.eval_end))


def _after(cutoff: pd.Timestamp, name: str) -> Split:
    """Train up to the end of cutoff's month, evaluate on the next HORIZON_MONTHS months."""
    train_end = cutoff + pd.offsets.MonthEnd(0)
    eval_start = train_end + pd.Timedelta(days=1)
    eval_end = eval_start + pd.offsets.MonthEnd(HORIZON_MONTHS)
    return Split(name, DATA_START, train_end, eval_start, eval_end)


FOLDS: tuple[Split, ...] = tuple(
    _after(pd.Timestamp(f"2025-{month:02d}-01"), f"Fold {i}") for i, month in enumerate(range(3, 7), start=1)
)
TEST = _after(pd.Timestamp("2025-08-01"), "Test")

assert all(fold.eval_end < TEST_START for fold in FOLDS), "validation folds must stay out of the test period"
assert TEST.eval_start == TEST_START and TEST.eval_end == DATA_END


def development(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Rows before the test period: the only data model decisions may look at."""
    return df[pd.to_datetime(df[date_col]) < TEST_START]


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, as a fraction (0.02 = 2%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true) / y_true))


def mae(y_true, y_pred) -> float:
    """Mean absolute error, in dollars."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))
