"""The two models compared in the report.

* baseline_model(): plain linear regression on log price, using the feature table as it is.
* final_model(): a regularised linear model on log price whose inputs follow the EDA:
  curves for distance and weight, a straight-line market index effect and upward
  drift, a quarter-end ramp sized separately for each equipment type, weekday,
  equipment, pickup and delivery city, and lane (city-pair) effects. A city never
  seen in training is priced with the average effect of its nearest known cities.

Both are scikit-learn estimators that take the table from freight.features and
return prices in dollars, so evaluation and prediction treat them the same way.
Modelling log price turns percentage effects (Reefer costs ~13% more at any
distance) into additive ones and matches the noise, which is ~2% of the price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

from freight.features import CATEGORICAL, NUMERIC

# Final-model settings, chosen on the walk-forward validation folds
RIDGE_ALPHA = 0.01      # light shrinkage overall...
LANE_SHRINK = 0.1       # ...and ~100x stronger for lane effects, which rest on ~10 loads each
DISTANCE_KNOTS = 8      # flexibility of the distance curve
WEIGHT_KNOTS = 6        # flexibility of the weight curve (S-shaped)
NEIGHBOURS = 3          # an unseen city borrows the average effect of its 3 nearest known cities

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]
FLAGS = ["weight_missing", "weight_capped", "market_index_missing"]
RAMPS = [f"quarter_end_ramp_{e.lower().replace(' ', '_')}" for e in EQUIPMENT]


def _log_price(regressor) -> TransformedTargetRegressor:
    """Fit on log price, predict in dollars."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def baseline_model() -> TransformedTargetRegressor:
    """Plain linear regression on log price, using the feature table as it is."""
    columns = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categories", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])
    return _log_price(make_pipeline(columns, LinearRegression()))


def final_model_inputs(X: pd.DataFrame) -> pd.DataFrame:
    """The final model's inputs, derived from the shared feature table."""
    out = pd.DataFrame(index=X.index)
    out["log_distance"] = X["log_distance"]
    out["weight_k"] = X["weight"] / 1000
    out[FLAGS] = X[FLAGS]
    out["market_index"] = X["market_index"]
    out["trend"] = X["days_since_start"] / 100
    for equipment, ramp in zip(EQUIPMENT, RAMPS):
        # the quarter-end surge is much larger for flatbeds than for dry vans
        out[ramp] = X["quarter_end_ramp"] * (X["equipment"] == equipment)
    out["weekday"] = X["day_of_week"].astype(str)
    for col in ["equipment", "pickup", "delivery"]:
        out[col] = X[col].astype(str)
    out["lane"] = out["pickup"] + " > " + out["delivery"]
    return out


def _shrink_lanes(lane_columns):
    # Ridge penalises coefficients on shrunk columns more: scaling by 0.1 means a 100x stronger penalty
    return lane_columns * LANE_SHRINK


def _final_pipeline() -> TransformedTargetRegressor:
    """Regularised linear model on log price with the EDA-informed inputs."""
    columns = ColumnTransformer([
        ("distance", SplineTransformer(n_knots=DISTANCE_KNOTS, extrapolation="linear"), ["log_distance"]),
        ("weight", SplineTransformer(n_knots=WEIGHT_KNOTS, extrapolation="linear"), ["weight_k"]),
        ("straight_line", "passthrough", FLAGS + ["market_index", "trend"] + RAMPS),
        ("categories", OneHotEncoder(drop="first", handle_unknown="ignore"), ["equipment", "weekday"]),
        ("cities", OneHotEncoder(handle_unknown="ignore"), ["pickup", "delivery"]),
        ("lanes", make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                FunctionTransformer(_shrink_lanes, accept_sparse=True, feature_names_out="one-to-one")),
         ["lane"]),
    ])
    return _log_price(make_pipeline(FunctionTransformer(final_model_inputs), columns, Ridge(alpha=RIDGE_ALPHA)))


class FinalModel(BaseEstimator, RegressorMixin):
    """The final model, able to price cities it never saw in training.

    City effects follow a smooth regional pattern, so a pickup or delivery city
    missing from the training data takes the average learned effect of the known
    cities nearest to it on the map. Loads between known cities are unaffected.
    """

    def __init__(self, neighbours: int = NEIGHBOURS):
        self.neighbours = neighbours

    def fit(self, X: pd.DataFrame, y) -> "FinalModel":
        self.model_ = _final_pipeline().fit(X, y)
        pipe = self.model_.regressor_
        coef = pd.Series(pipe[-1].coef_, index=pipe[1].get_feature_names_out())
        self.cities_ = {}  # per side: each known city's map position and learned effect on log price
        for side in ("pickup", "delivery"):
            prefix = f"cities__{side}_"
            effects = coef[coef.index.str.startswith(prefix)].rename(lambda name: name[len(prefix):])
            positions = X[[f"{side}_lat", f"{side}_lon"]].groupby(X[side].astype(str).to_numpy()).first()
            positions.columns = ["lat", "lon"]
            self.cities_[side] = positions.join(effects.rename("effect"), how="inner")
        return self

    def unseen_city_adjustment(self, X: pd.DataFrame) -> np.ndarray:
        """Log-price adjustment for loads whose pickup or delivery city was not in the training data."""
        adjustment = np.zeros(len(X))
        for side, known in self.cities_.items():
            lat, lon = X[f"{side}_lat"].to_numpy(float), X[f"{side}_lon"].to_numpy(float)
            unseen = ~X[side].astype(str).isin(known.index).to_numpy() & ~np.isnan(lat) & ~np.isnan(lon)
            if unseen.any():
                distance = np.hypot(lat[unseen, None] - known["lat"].to_numpy(), lon[unseen, None] - known["lon"].to_numpy())
                nearest = np.argsort(distance, axis=1)[:, :self.neighbours]
                adjustment[unseen] += known["effect"].to_numpy()[nearest].mean(axis=1)
        return adjustment

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X) * np.exp(self.unseen_city_adjustment(X))


def final_model() -> FinalModel:
    """The final model (see FinalModel)."""
    return FinalModel()
