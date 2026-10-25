# Freight Rate Prediction

Solution to the Spotter Machine Learning Engineer assessment: predict the posted rate of a freight load from its lane, equipment, weight, date and market signals.

The original assessment instructions are in [`docs/assessment_readme.md`](docs/assessment_readme.md).

## Repository layout

```
.
├── data/
│   ├── train_test.csv                       # labelled development data (48,000 loads)
│   ├── validation.csv                       # 12,000 loads to predict (no labels)
│   ├── validation_predictions_template.csv  # load_id list to fill with predictions
│   └── december_chart_inputs.csv            # 31 fixed-lane December rows to fill
├── docs/
│   └── assessment_readme.md                 # instructions provided with the assessment
├── freight/
│   ├── cleaning.py                          # data-quality fixes and corrupted-price screen
│   ├── data.py                              # loading the assessment files
│   ├── evaluate.py                          # scoring models on the folds and the test period
│   ├── features.py                          # feature table shared by all models
│   ├── models.py                            # baseline and final model
│   ├── report_figures.py                    # report figures not produced by a notebook
│   ├── style.py                             # shared figure style
│   └── validation.py                        # time-based validation split and accuracy measures
├── notebooks/
│   ├── 01_eda.ipynb                         # exploratory data analysis (labelled data only)
│   └── 02_modelling.ipynb                   # baseline vs final model, and what the final model learned
├── reports/
│   ├── eda_findings.md                      # summary of EDA findings
│   └── figures/
│       ├── eda/                             # EDA figures
│       ├── model/                           # model comparison figures
│       └── validation/                      # validation design figure
├── tests/                                   # automated checks of cleaning, features and splits
├── score.py                                 # provided scorer: validates outputs, draws December chart
├── pytest.ini
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Exploratory data analysis

Findings are summarised in [`reports/eda_findings.md`](reports/eda_findings.md). To re-run the notebook and regenerate the figures:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## Validation design

Spotter's loads (November–December 2025) come after the labelled data, so validation is time-based ([`freight/validation.py`](freight/validation.py)):

- **Model choices** use four walk-forward folds inside January–August. Each fold trains on everything up to a cut-off and is evaluated on the following two months (April–May, May–June, June–July, July–August).
- **September–October** is a test period, scored once after the model is chosen.
- **The final model** is retrained on all of January–October.

Accuracy is measured as mean absolute percentage error (MAPE), with mean absolute error in dollars alongside. To redraw the timeline figure:

```bash
python -m freight.report_figures
```

## Data cleaning and features

The fixes follow the EDA findings ([`freight/cleaning.py`](freight/cleaning.py)):

- **Corrupted prices** (about 1.4% of loads, scaled up or down by a factor of 2–6) are flagged by comparing each posted rate with what comparable loads cost, and are excluded from training and from scoring.
- **Negative weights** are sign flips and are made positive; missing and capped (47,500 lb) weights are flagged.
- **Missing market index** values are filled with the average of the same day's loads.

[`freight/features.py`](freight/features.py) builds one feature table for every model: distance, equipment, weight, market index, pickup and delivery city with coordinates, and calendar features (time trend, weekday, and a quarter-end ramp that builds over the last five weeks of each quarter). Anything learned from data, including the corrupted-price screen, is fitted on each split's training rows only.

`quote_signal` is deliberately left out: its relationship with price switches on and off from month to month, and in walk-forward validation a model using it mispriced a whole two-month fold by about 12%.

## Models

[`freight/models.py`](freight/models.py) defines the two models. Both are trained on log price and return dollars.

- **Baseline:** plain linear regression on the feature table as it is.
- **Final model:** a regularised linear model whose inputs follow the EDA:
  - smooth curves for distance and weight;
  - a straight-line market index effect and upward drift;
  - a quarter-end ramp sized separately for each equipment type;
  - weekday, equipment, pickup and delivery city;
  - lane effects, shrunk strongly because each lane has only ~10 loads.

[`notebooks/02_modelling.ipynb`](notebooks/02_modelling.ipynb) compares them on the walk-forward folds and shows what the final model learned. The comparison also runs from the command line:

```bash
python -m freight.evaluate
python -m nbconvert --to notebook --execute --inplace notebooks/02_modelling.ipynb
```

## Tests

```bash
python -m pytest
```

## Scoring the outputs

Once `validation_predictions.csv` and the `predicted_rate` column of `data/december_chart_inputs.csv` are filled:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

This checks both files and writes `scorer_results/candidate_december.png`.
