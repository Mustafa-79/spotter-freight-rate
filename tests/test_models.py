import numpy as np
import pytest

from freight.features import FeatureBuilder
from freight.models import baseline_model, final_model
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
