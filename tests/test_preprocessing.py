import numpy as np
import pandas as pd
import pytest

from freight.cleaning import LabelScreen, fill_market_index, fix_weights
from freight.features import CATEGORICAL, NUMERIC, FeatureBuilder, calendar_features
from freight.validation import FOLDS, TEST, TEST_START, mape


def synthetic_loads(n=3000, seed=0):
    """Loads priced as a fixed charge plus a per-mile rate, with 2% multiplicative noise."""
    rng = np.random.default_rng(seed)
    cities = pd.DataFrame({"city": list("ABCDE"), "lat": [30, 33, 36, 39, 42], "lon": [-100, -95, -90, -85, -80]})
    pick = cities.sample(n, replace=True, random_state=1).reset_index(drop=True)
    drop = cities.sample(n, replace=True, random_state=2).reset_index(drop=True)
    equipment = rng.choice(["Dry Van", "Reefer", "Flatbed"], n)
    distance = rng.uniform(100, 3000, n).round(1)
    premium = pd.Series(equipment).map({"Dry Van": 1.0, "Reefer": 1.13, "Flatbed": 1.08}).to_numpy()
    rate = (250 + 1.9 * distance) * premium * np.exp(rng.normal(0, 0.02, n))
    return pd.DataFrame({
        "pickup": pick.city, "pickup_lat": pick.lat, "pickup_lon": pick.lon,
        "delivery": drop.city, "delivery_lat": drop.lat, "delivery_lon": drop.lon,
        "distance": distance, "equipment": equipment, "weight": rng.uniform(10_000, 45_000, n).round(),
        "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D"),
        "market_index": rng.normal(1.0, 0.02, n), "quote_signal": rng.normal(2.0, 0.3, n),
        "posted_rate": rate.round(2),
    })


def test_fix_weights_flips_signs_and_flags_cap_and_gaps():
    out = fix_weights(pd.DataFrame({"weight": [-30_000, 47_500, np.nan, 20_000]}))
    assert out["weight"].tolist()[:2] == [30_000, 47_500] and np.isnan(out["weight"].iloc[2])
    assert out["weight_capped"].tolist() == [0, 1, 0, 0]
    assert out["weight_missing"].tolist() == [0, 0, 1, 0]


def test_fill_market_index_uses_same_day_then_neighbouring_days():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02", "2025-01-03"]),
        "market_index": [1.0, np.nan, 1.2, np.nan, 1.3],
    })
    out = fill_market_index(df)
    assert out["market_index"].tolist() == pytest.approx([1.0, 1.1, 1.2, 1.2, 1.3])
    assert out["market_index_missing"].tolist() == [0, 1, 0, 1, 0]


def test_label_screen_flags_exactly_the_scaled_prices():
    df = synthetic_loads()
    corrupted = np.zeros(len(df), dtype=bool)
    corrupted[:30], corrupted[30:60] = True, True
    df.loc[:29, "posted_rate"] *= 3
    df.loc[30:59, "posted_rate"] /= 3
    flags = LabelScreen().fit(df).flag(df)
    assert (flags.to_numpy() == corrupted).all()


def test_quarter_end_ramp_covers_the_last_five_weeks_and_accelerates():
    dates = pd.Series(pd.to_datetime(["2025-02-24", "2025-03-10", "2025-03-31", "2025-04-01", "2025-12-31"]))
    ramp = calendar_features(dates)["quarter_end_ramp"]
    # 35 days out: 0; 21 days out: (14/35)^1.5; last day of a quarter: 1; first day of the next: 0
    assert ramp.tolist() == pytest.approx([0, (14 / 35) ** 1.5, 1, 0, 1])


def test_feature_builder_fills_columns_missing_from_new_rows():
    train = synthetic_loads()
    builder = FeatureBuilder().fit(train)
    new = train[["pickup", "delivery", "distance", "equipment", "weight", "date"]].head(5)
    daily = pd.Series(1.05, index=pd.date_range("2024-12-01", "2025-04-30"))
    X = builder.transform(new, daily_market_index=daily)
    assert list(X.columns) == NUMERIC + CATEGORICAL
    assert not X.isna().any().any()
    assert (X["market_index"] == 1.05).all()


def test_quote_signal_is_not_a_feature():
    X = FeatureBuilder().fit(synthetic_loads()).transform(synthetic_loads().head(5))
    assert "quote_signal" not in X.columns


def test_validation_folds_stay_before_the_test_period_without_overlap():
    dates = pd.Series(pd.date_range("2025-01-01", "2025-10-31"))
    for fold in FOLDS:
        train, evaluate = fold.masks(dates)
        assert not (train & evaluate).any()
        assert dates[train].max() < dates[evaluate].min()
        assert dates[evaluate].max() < TEST_START
    train, evaluate = TEST.masks(dates)
    assert dates[train].max() < TEST_START == dates[evaluate].min()


def test_mape():
    assert mape([100, 200], [110, 180]) == pytest.approx(0.10)
