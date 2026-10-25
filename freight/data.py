"""Loading the assessment files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TARGET = "posted_rate"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, parse_dates=["date"])


def load_labelled() -> pd.DataFrame:
    """The 48,000 labelled loads (train_test.csv), January–October 2025."""
    return _read("train_test.csv")


def load_prediction_set() -> pd.DataFrame:
    """Spotter's 12,000 unlabelled loads (validation.csv), November–December 2025."""
    return _read("validation.csv")


def load_december_inputs() -> pd.DataFrame:
    """The 31 fixed-lane December rows behind the scorer's chart."""
    return _read("december_chart_inputs.csv")
