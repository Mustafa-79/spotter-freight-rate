"""Scoring the models on the walk-forward folds and on the test period.

Run from the repository root:
    python -m freight.evaluate          # the four walk-forward folds (Jan–Aug)
    python -m freight.evaluate --test   # the September–October test period
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from freight.data import load_labelled
from freight.features import prepare_split
from freight.models import baseline_model, final_model
from freight.validation import FOLDS, TEST, Split, mae, mape

MODELS: dict[str, Callable] = {
    "Baseline: linear regression": baseline_model,
    "Final model": final_model,
}


def predict_splits(df: pd.DataFrame, splits: tuple[Split, ...], models: dict[str, Callable] = MODELS) -> pd.DataFrame:
    """Fit every model on each split's training rows and predict its evaluation rows.

    Returns one row per evaluated load and model, so results can be broken down
    any way (by fold, equipment, distance, weeks ahead, ...).
    """
    frames = []
    for split in splits:
        data = prepare_split(df, split)
        info = df.loc[data.X_eval.index, ["date", "equipment", "distance"]]
        for name, make_model in models.items():
            model = make_model().fit(data.X_train, data.y_train)
            frames.append(info.assign(split=split.name, model=name, actual=data.y_eval,
                                      predicted=model.predict(data.X_eval)))
    return pd.concat(frames, ignore_index=False).rename_axis("row").reset_index()


def summarise(predictions: pd.DataFrame, by=("model", "split")) -> pd.DataFrame:
    """MAPE, dollar error and bias (average signed % error) for each group."""
    def metrics(g):
        return pd.Series({
            "MAPE": mape(g.actual, g.predicted),
            "MAE ($)": mae(g.actual, g.predicted),
            "bias": float(np.mean(g.predicted / g.actual - 1)),
            "loads": len(g),
        })
    return predictions.groupby(list(by), sort=False, observed=True).apply(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", action="store_true", help="score the September–October test period")
    args = parser.parse_args()
    splits = (TEST,) if args.test else FOLDS
    table = summarise(predict_splits(load_labelled(), splits))
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table)
    if not args.test:
        print(table.groupby(level="model", sort=False)[["MAPE", "MAE ($)", "bias"]].mean())


if __name__ == "__main__":
    main()
