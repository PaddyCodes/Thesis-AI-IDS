from pathlib import Path

import joblib
import pandas as pd
import pytest


MODEL_FILE = Path(
    "outputs/models/random_forest.joblib"
)

FEATURE_FILE = Path(
    "outputs/models/random_forest_features.txt"
)

VALIDATION_FILE = Path(
    "data/processed/splits/validation.csv"
)


@pytest.fixture(scope="session")
def saved_model():
    assert MODEL_FILE.exists(), (
        "Saved Random Forest model not found."
    )

    return joblib.load(MODEL_FILE)


@pytest.fixture(scope="session")
def saved_features():
    assert FEATURE_FILE.exists(), (
        "Saved feature list not found."
    )

    return FEATURE_FILE.read_text(
        encoding="utf-8"
    ).splitlines()


@pytest.fixture(scope="session")
def validation_sample(saved_features):
    assert VALIDATION_FILE.exists(), (
        "Validation split not found."
    )

    # Only load a small sample here. We are testing
    # the saved artefact rather than re-running validation.
    df = pd.read_csv(
        VALIDATION_FILE,
        nrows=100,
    )

    return df[saved_features]


def test_saved_model_loads(saved_model):
    assert saved_model is not None


def test_saved_feature_count(saved_features):
    assert len(saved_features) == 69


def test_saved_features_are_unique(saved_features):
    assert len(saved_features) == len(
        set(saved_features)
    )


def test_model_feature_count(
    saved_model,
    saved_features,
):
    assert saved_model.n_features_in_ == len(
        saved_features
    )


def test_model_has_expected_parameters(
    saved_model,
):
    parameters = saved_model.get_params()

    assert parameters[
        "n_estimators"
    ] == 100

    assert parameters[
        "max_depth"
    ] == 20

    assert parameters[
        "min_samples_split"
    ] == 2

    assert parameters[
        "min_samples_leaf"
    ] == 2

    assert parameters[
        "max_features"
    ] == 0.5

    assert parameters[
        "class_weight"
    ] == "balanced_subsample"

    assert parameters[
        "random_state"
    ] == 42


def test_saved_model_can_predict(
    saved_model,
    validation_sample,
):
    predictions = saved_model.predict(
        validation_sample
    )

    assert len(predictions) == 100

    assert set(predictions).issubset(
        {0, 1}
    )


def test_saved_model_can_return_probabilities(
    saved_model,
    validation_sample,
):
    probabilities = (
        saved_model.predict_proba(
            validation_sample
        )
    )

    assert probabilities.shape == (
        100,
        2,
    )

    assert (
        probabilities >= 0
    ).all()

    assert (
        probabilities <= 1
    ).all()