"""Data-quality fixes established in the EDA (notebooks/01_eda.ipynb).

* Negative weights are sign flips; 47,500 lb is a recording cap.
* market_index is essentially a daily value, so a missing value is recovered from
  the same day's other loads.
* About 1.4% of posted rates are scaled up or down by a factor of 2–6. Every other
  load sits within ~10% of what comparable loads cost, so those rows are found
  reliably and excluded from training and scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from freight.data import TARGET

WEIGHT_CAP = 47_500
CORRUPTION_RATIO = 1.5  # normal loads stay within ~1.1x of expected, corrupted ones are 2x or more away
SERIES_START = pd.Timestamp("2025-01-01")


def fix_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Undo sign flips and flag missing and capped weights. Missing weights stay NaN."""
    out = df.copy()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_capped"] = (out["weight"] == WEIGHT_CAP).astype(int)
    return out


def fill_market_index(df: pd.DataFrame, daily: pd.Series | None = None) -> pd.DataFrame:
    """Fill missing market_index with the average of the same day's loads.

    Days with no index at all in df take it from `daily` (date -> index) when given;
    any remaining gaps are interpolated between neighbouring days.
    """
    out = df.copy()
    if "market_index" not in out:
        out["market_index"] = np.nan
    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out.groupby("date")["market_index"].transform("mean"))
    if out["market_index"].isna().any():
        known = out.groupby("date")["market_index"].mean().dropna()
        if daily is not None:
            known = known.combine_first(daily.dropna())
        if known.empty:  # nothing to fill from
            return out
        days = pd.date_range(min(known.index.min(), out["date"].min()), max(known.index.max(), out["date"].max()))
        by_day = known.reindex(days).interpolate(limit_direction="both")
        out["market_index"] = out["market_index"].fillna(out["date"].map(by_day))
    return out


class LabelScreen:
    """Flags posted rates that are far from what comparable loads cost.

    The expected rate comes from a linear fit on log price (distance and weight
    curves, equipment, market index, pickup and delivery city, linear time). It is
    fitted twice, the second time without loads more than `ratio` away, so the
    corrupted rows cannot pull it. Fit on training rows only, then flag any period.
    """

    def __init__(self, ratio: float = CORRUPTION_RATIO):
        self.ratio = ratio

    @staticmethod
    def _reference_model():
        columns = ColumnTransformer([
            ("distance", SplineTransformer(n_knots=6, extrapolation="linear"), ["log_distance"]),
            ("weight", SplineTransformer(n_knots=5, extrapolation="linear"), ["weight_k"]),
            ("numeric", "passthrough", ["market_index", "days_since_start"]),
            ("categories", OneHotEncoder(handle_unknown="ignore"), ["equipment", "pickup", "delivery"]),
        ])
        return make_pipeline(columns, Ridge(alpha=1.0))

    def _inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = fill_market_index(fix_weights(df))
        return pd.DataFrame({
            "log_distance": np.log(clean["distance"]),
            "weight_k": clean["weight"].fillna(self.weight_median_) / 1000,
            "market_index": clean["market_index"],
            "days_since_start": (clean["date"] - SERIES_START).dt.days,
            "equipment": clean["equipment"],
            "pickup": clean["pickup"],
            "delivery": clean["delivery"],
        }, index=df.index)

    def fit(self, df: pd.DataFrame) -> "LabelScreen":
        self.weight_median_ = df["weight"].abs().median()
        X, y = self._inputs(df), np.log(df[TARGET])
        first = self._reference_model().fit(X, y)
        keep = np.abs(y - first.predict(X)) < np.log(self.ratio)
        self.model_ = self._reference_model().fit(X[keep], y[keep])
        return self

    def expected_rate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.exp(self.model_.predict(self._inputs(df))), index=df.index)

    def flag(self, df: pd.DataFrame) -> pd.Series:
        """True where the posted rate is more than `ratio` times above or below the expected rate."""
        log_ratio = np.log(df[TARGET] / self.expected_rate(df))
        return log_ratio.abs() > np.log(self.ratio)
