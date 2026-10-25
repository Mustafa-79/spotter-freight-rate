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
│   ├── report_figures.py                    # report figures not produced by a notebook
│   ├── style.py                             # shared figure style
│   └── validation.py                        # time-based validation split and accuracy measures
├── notebooks/
│   └── 01_eda.ipynb                         # exploratory data analysis (labelled data only)
├── reports/
│   ├── eda_findings.md                      # summary of EDA findings
│   └── figures/
│       ├── eda/                             # EDA figures
│       └── validation/                      # validation design figure
├── score.py                                 # provided scorer: validates outputs, draws December chart
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

## Scoring the outputs

Once `validation_predictions.csv` and the `predicted_rate` column of `data/december_chart_inputs.csv` are filled:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

This checks both files and writes `scorer_results/candidate_december.png`.
