import json
from pathlib import Path

import numpy as np
import pytest

from model import CreditRiskPredictor

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((ROOT / "tests" / "fixtures.json").read_text())


@pytest.fixture(scope="module")
def predictor():
    return CreditRiskPredictor(ROOT / "artifacts")


def test_artifacts_load(predictor):
    assert len(predictor.features) == 23
    assert predictor.mean.shape == predictor.scale.shape == (23,)


@pytest.mark.parametrize("row", FIXTURES["rows"])
def test_matches_module8_predictions(predictor, row):
    p, label = predictor.predict_one(row["features"])
    assert p == pytest.approx(row["probability"], abs=1e-4)
    assert label == row["label"]


def test_probabilities_are_valid(predictor):
    X = np.array([list(r["features"].values()) for r in FIXTURES["rows"]])
    p = predictor.predict_proba(X)
    assert np.all((p >= 0) & (p <= 1))


def test_late_payer_scores_riskier(predictor):
    base = dict(FIXTURES["rows"][-1]["features"])
    late = dict(base, **{k: 3 for k in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]})
    assert predictor.predict_one(late)[0] > predictor.predict_one(base)[0]


def test_wrong_feature_count_rejected(predictor):
    with pytest.raises(ValueError):
        predictor.predict_proba(np.zeros((1, 5)))


def test_gradio_app_builds():
    import app

    p, verdict = app.score(50000, 2, 2, 2, 35, *[0] * 6, *[20000] * 6, *[2000] * 6)
    assert set(p) == {"Default", "No default"}
    assert "Likely" in verdict
