# Freight Rate Prediction

Predicts the posted rate of freight loads for Spotter's Machine Learning Engineer assessment. The final model is a regularised linear regression on log price, chosen on walk-forward validation folds and scored once on held-back months. The full write-up is in [`reports/report.pdf`](reports/report.pdf).

| MAPE (average % error) | Walk-forward folds (Jan–Aug) | Test (Sep–Oct) |
|---|---|---|
| Baseline: linear regression | 1.73% | 1.70% |
| Final model | 1.37% | 1.41% |

## Run it

```bash
python -m pip install -r requirements.txt
python -m freight.predict
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`freight.predict` trains the final model on all labelled data and writes `validation_predictions.csv` and the `predicted_rate` column of `data/december_chart_inputs.csv`. `score.py` checks both files and draws `scorer_results/candidate_december.png`.

To reproduce the results above and run the tests:

```bash
python -m freight.evaluate          # walk-forward folds
python -m freight.evaluate --test   # September–October test
python -m pytest
```

## Where things are

- `freight/`: cleaning, features, validation split, models and prediction
- `notebooks/01_eda.ipynb`: exploratory analysis; `notebooks/02_modelling.ipynb`: model comparison and test results
- `reports/`: the report (`report.pdf`, built from `report.tex` with XeLaTeX, e.g. `tectonic reports/report.tex`), EDA summary and figures
- `data/`: the assessment files; `docs/`: the original instructions
- `tests/`: checks for cleaning, features, models and splits
