"""Train the final model on all labelled data and write the submission files.

Run from the repository root:
    python -m freight.predict

Writes:
* validation_predictions.csv: load_id,predicted_rate for the 12,000 loads in validation.csv,
  in the order of data/validation_predictions_template.csv;
* data/december_chart_inputs.csv: its predicted_rate column filled for the 31 December days.

The December chart rows carry no market index, so each day takes the average
market index of that day's loads in validation.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from freight.cleaning import LabelScreen
from freight.data import DATA_DIR, ROOT, TARGET, load_december_inputs, load_labelled, load_prediction_set
from freight.features import FeatureBuilder
from freight.models import final_model

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"


def train(labelled: pd.DataFrame):
    """Remove corrupted prices, then fit the feature builder and the final model on what remains."""
    clean = labelled[~LabelScreen().fit(labelled).flag(labelled)]
    builder = FeatureBuilder().fit(clean)
    model = final_model().fit(builder.transform(clean), clean[TARGET])
    return builder, model, clean


def daily_market_index(loads: pd.DataFrame) -> pd.Series:
    """Average market index of each day's loads."""
    return loads.groupby("date")["market_index"].mean()


def fill_template(template: pd.DataFrame, load_ids: pd.Series, predicted: np.ndarray) -> pd.DataFrame:
    """The template's load_ids, in its order, with their predicted rates."""
    by_id = pd.Series(np.round(predicted, 2), index=load_ids.to_numpy())
    return pd.DataFrame({"load_id": template["load_id"], "predicted_rate": template["load_id"].map(by_id)})


def report_checks(clean: pd.DataFrame, loads: pd.DataFrame, X: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Print sanity checks on the prediction set and the predictions."""
    seen = set(clean["pickup"]) | set(clean["delivery"])
    lanes = set(zip(clean["pickup"], clean["delivery"]))
    unseen_city = ~loads["pickup"].isin(seen) | ~loads["delivery"].isin(seen)
    unseen_lane = ~pd.Series(list(zip(loads["pickup"], loads["delivery"])), index=loads.index).isin(lanes)
    rate_per_mile = predictions["predicted_rate"].to_numpy() / loads.set_index("load_id").loc[predictions["load_id"], "distance"].to_numpy()
    train_rpm = clean[TARGET] / clean["distance"]
    print(f"Prediction set: {len(loads):,} loads, {loads['date'].min():%Y-%m-%d} to {loads['date'].max():%Y-%m-%d}")
    print(f"  loads with a city not seen in training: {int(unseen_city.sum())}; on a lane not seen in training: "
          f"{int(unseen_lane.sum())}")
    print(f"  negative weights fixed: {int((loads['weight'] < 0).sum())}; missing weights: {int(loads['weight'].isna().sum())}; "
          f"missing market index filled: {int(X['market_index_missing'].sum())}")
    print(f"  market index range {X['market_index'].min():.2f}–{X['market_index'].max():.2f} "
          f"(training {clean['market_index'].min():.2f}–{clean['market_index'].max():.2f})")
    print(f"  predicted rate ${predictions['predicted_rate'].min():,.0f}–${predictions['predicted_rate'].max():,.0f}, "
          f"median ${predictions['predicted_rate'].median():,.0f}; rate per mile {rate_per_mile.min():.2f}–"
          f"{rate_per_mile.max():.2f} (training 1st–99th percentile {train_rpm.quantile(0.01):.2f}–"
          f"{train_rpm.quantile(0.99):.2f})")


def main() -> None:
    labelled = load_labelled()
    builder, model, clean = train(labelled)
    print(f"Trained on {len(clean):,} labelled loads ({len(labelled) - len(clean)} corrupted prices removed)")

    loads = load_prediction_set()
    X = builder.transform(loads)
    template = pd.read_csv(TEMPLATE_PATH)
    predictions = fill_template(template, loads["load_id"], model.predict(X))
    assert list(predictions.columns) == ["load_id", "predicted_rate"] and len(predictions) == len(template)
    assert predictions["predicted_rate"].notna().all() and (predictions["predicted_rate"] > 0).all()
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    report_checks(clean, loads, X, predictions)
    print(f"Wrote {PREDICTIONS_PATH.name} ({len(predictions):,} rows)")

    december = load_december_inputs()
    X_dec = builder.transform(december.drop(columns="predicted_rate"), daily_market_index=daily_market_index(loads))
    december["predicted_rate"] = np.round(model.predict(X_dec), 2)
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december.to_csv(DECEMBER_PATH, index=False)
    print(f"Filled {DECEMBER_PATH.relative_to(ROOT).as_posix()}: ${december['predicted_rate'].min():,.0f}–"
          f"${december['predicted_rate'].max():,.0f} across December "
          f"(market index {X_dec['market_index'].min():.2f}–{X_dec['market_index'].max():.2f})")


if __name__ == "__main__":
    main()
