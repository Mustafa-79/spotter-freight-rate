"""Model features built from the cleaned data.

Everything learned from data (fill values, city coordinates, category lists) is
learned in FeatureBuilder.fit() from training rows only, so the same builder can
be applied to validation, test and Spotter's loads without leaking information.

quote_signal is deliberately left out: its relationship with price switches on
and off from month to month, and a model that relies on it mispriced a whole
validation fold by about 12% when the relationship shifted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from freight.cleaning import SERIES_START, LabelScreen, fill_market_index, fix_weights
from freight.data import TARGET
from freight.validation import Split

QUARTER_END_WINDOW = 35  # days: prices start rising about five weeks before a quarter ends...
QUARTER_END_SHAPE = 1.5  # ...and the rise accelerates into the last day

NUMERIC = [
    "distance", "log_distance",
    "weight", "weight_missing", "weight_capped",
    "market_index", "market_index_missing",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "days_since_start", "day_of_week", "days_to_quarter_end", "quarter_end_ramp",
]
CATEGORICAL = ["equipment", "pickup", "delivery"]


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Time trend, weekday and position relative to the quarter end."""
    dates = pd.to_datetime(dates)
    days_to_qend = ((dates + pd.offsets.QuarterEnd(0)) - dates).dt.days
    return pd.DataFrame({
        "days_since_start": (dates - SERIES_START).dt.days,
        "day_of_week": dates.dt.dayofweek,
        "days_to_quarter_end": days_to_qend,
        # 0 until five weeks before a quarter ends, then rising, ever faster, to 1 on its last day
        "quarter_end_ramp": ((QUARTER_END_WINDOW - days_to_qend) / QUARTER_END_WINDOW).clip(0, 1) ** QUARTER_END_SHAPE,
    }, index=dates.index)


class FeatureBuilder:
    """Turns raw load rows into the feature table used by every model."""

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.weight_median_ = df["weight"].abs().median()
        stops = pd.concat([
            df[[f"{side}", f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
            for side in ("pickup", "delivery")
        ])
        self.city_coords_ = stops.groupby("city")[["lat", "lon"]].first()
        self.categories_ = {
            "equipment": sorted(df["equipment"].unique()),
            "pickup": list(self.city_coords_.index),
            "delivery": list(self.city_coords_.index),
        }
        return self

    def transform(self, df: pd.DataFrame, daily_market_index: pd.Series | None = None) -> pd.DataFrame:
        """Feature table for df. `daily_market_index` fills days whose loads carry no market_index."""
        clean = fill_market_index(fix_weights(df), daily_market_index)
        out = pd.DataFrame(index=df.index)
        out["distance"] = clean["distance"]
        out["log_distance"] = np.log(clean["distance"])
        out["weight"] = clean["weight"].fillna(self.weight_median_)
        out["weight_missing"] = clean["weight_missing"]
        out["weight_capped"] = clean["weight_capped"]
        out["market_index"] = clean["market_index"]
        out["market_index_missing"] = clean["market_index_missing"]
        for side in ("pickup", "delivery"):
            for axis in ("lat", "lon"):
                lookup = clean[side].map(self.city_coords_[axis])
                given = clean[f"{side}_{axis}"] if f"{side}_{axis}" in clean else lookup
                out[f"{side}_{axis}"] = given.fillna(lookup)
        out = out.join(calendar_features(clean["date"]))
        for col, categories in self.categories_.items():
            out[col] = pd.Categorical(clean[col], categories=categories)
        return out[NUMERIC + CATEGORICAL]


@dataclass
class SplitData:
    """Features and targets for one train/evaluate split, corrupted prices removed."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_eval: pd.DataFrame
    y_eval: pd.Series
    flagged_train: int
    flagged_eval: int


def prepare_split(df: pd.DataFrame, split: Split) -> SplitData:
    """Screen labels and build features using the split's training rows only."""
    train_mask, eval_mask = split.masks(df["date"])
    train, evaluate = df[train_mask], df[eval_mask]
    screen = LabelScreen().fit(train)
    bad_train, bad_eval = screen.flag(train), screen.flag(evaluate)
    train, evaluate = train[~bad_train], evaluate[~bad_eval]
    builder = FeatureBuilder().fit(train)
    return SplitData(builder.transform(train), train[TARGET], builder.transform(evaluate), evaluate[TARGET],
                     int(bad_train.sum()), int(bad_eval.sum()))
