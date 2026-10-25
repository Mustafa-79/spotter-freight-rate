import numpy as np
import pytest

from freight.features import FeatureBuilder
from freight.models import FinalModel, baseline_model, final_model
from freight.validation import mape
from test_preprocessing import synthetic_loads


@pytest.mark.parametrize("make_model", [baseline_model, final_model])
def test_models_learn_synthetic_prices_and_return_dollars(make_model):
    loads = synthetic_loads(n=4000)
    train, held_out = loads.iloc[:3000], loads.iloc[3000:]
    builder = FeatureBuilder().fit(train)
    model = make_model().fit(builder.transform(train), train["posted_rate"])
    predicted = model.predict(builder.transform(held_out))
    assert len(predicted) == len(held_out) and (predicted > 0).all()
    assert mape(held_out["posted_rate"], predicted) < 0.05  # the synthetic prices carry 2% noise


def test_final_model_prices_a_city_it_has_never_seen():
    train = synthetic_loads()
    builder = FeatureBuilder().fit(train)
    model = final_model().fit(builder.transform(train), train["posted_rate"])
    predicted = model.predict(builder.transform(train.head(3).assign(pickup="Unseen City")))
    assert np.isfinite(predicted).all() and (predicted > 0).all()


def test_unseen_city_is_priced_like_its_nearest_known_city():
    train = synthetic_loads()
    pricier = (train["pickup"] == "E") | (train["delivery"] == "E")
    train.loc[pricier, "posted_rate"] *= 1.05  # city E (lat 42, lon -80) is 5% pricier; city A sits at (30, -100)
    builder = FeatureBuilder().fit(train)
    model = FinalModel(neighbours=1).fit(builder.transform(train), train["posted_rate"])
    load = train[(train["pickup"] == "B") & (train["delivery"] == "C")].head(1)
    next_to_e = load.assign(pickup="New City", pickup_lat=41.9, pickup_lon=-80.1)
    next_to_a = load.assign(pickup="New City", pickup_lat=30.1, pickup_lon=-99.9)
    ratio = model.predict(builder.transform(next_to_e))[0] / model.predict(builder.transform(next_to_a))[0]
    assert ratio == pytest.approx(1.05, abs=0.02)
