"""The two models compared in the report.

* baseline_model(): plain linear regression on log price, using the feature table as it is.
* final_model(): a regularised linear model on log price whose inputs follow the EDA:
  curves for distance and weight, a straight-line market index effect and upward
  drift, a quarter-end ramp sized separately for each equipment type, weekday,
  equipment, pickup and delivery city, and lane (city-pair) effects.

Both are scikit-learn estimators that take the table from freight.features and
return prices in dollars, so evaluation and prediction treat them the same way.
Modelling log price turns percentage effects (Reefer costs ~13% more at any
distance) into additive ones and matches the noise, which is ~2% of the price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
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


def final_model() -> TransformedTargetRegressor:
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
