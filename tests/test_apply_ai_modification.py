import numpy as np
import pandas as pd
import pytest

from src.apply_ai_modification import (
    transform_frame,
    validate_plan,
)


@pytest.fixture
def config():
    return {
        "generation": {
            "random_seed": 42,
            "minimum_scale": 0.5,
            "maximum_scale": 2.0,
            "maximum_flow_duration": 1200,
        },
        "timing_features": [
            "Flow Duration",
            "Flow IAT Mean",
        ],
        "inverse_rate_features": [
            "Flow Bytes/s",
            "Flow Packets/s",
        ],
    }


@pytest.fixture
def plan():
    return {
        "plan_version": 1,
        "strategy":
            "detector_independent_"
            "timing_morphing",
        "families": [
            {
                "label": "Attack A",
                "min_scale": 0.5,
                "max_scale": 0.5,
                "rationale":
                    "Synthetic test interval.",
            },
            {
                "label": "Attack B",
                "min_scale": 2.0,
                "max_scale": 2.0,
                "rationale":
                    "Synthetic test interval.",
            },
        ],
    }


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "Label": [
                "BENIGN",
                "Attack A",
                "Attack B",
            ],
            "binary_target": [
                0,
                1,
                1,
            ],
            "Destination Port": [
                443,
                80,
                22,
            ],
            "Flow Duration": [
                100.0,
                200.0,
                400.0,
            ],
            "Flow IAT Mean": [
                10.0,
                20.0,
                40.0,
            ],
            "Flow Bytes/s": [
                1000.0,
                500.0,
                250.0,
            ],
            "Flow Packets/s": [
                100.0,
                50.0,
                25.0,
            ],
            "Total Fwd Packets": [
                10,
                20,
                30,
            ],
        }
    )


def test_valid_plan(
    config,
    plan,
):
    validate_plan(
        plan,
        config,
        {
            "Attack A",
            "Attack B",
        },
    )


def test_plan_rejects_missing_family(
    config,
    plan,
):
    with pytest.raises(
        ValueError,
        match="coverage mismatch",
    ):
        validate_plan(
            plan,
            config,
            {
                "Attack A",
                "Attack B",
                "Attack C",
            },
        )


def test_benign_record_unchanged(
    config,
    plan,
    sample_frame,
):
    transformed, *_ = (
        transform_frame(
            sample_frame,
            config,
            plan,
        )
    )

    pd.testing.assert_series_equal(
        transformed.iloc[0],
        sample_frame.iloc[0],
        check_dtype=False,
    )


def test_non_timing_features_unchanged(
    config,
    plan,
    sample_frame,
):
    transformed, *_ = (
        transform_frame(
            sample_frame,
            config,
            plan,
        )
    )

    pd.testing.assert_series_equal(
        transformed[
            "Destination Port"
        ],
        sample_frame[
            "Destination Port"
        ],
    )

    pd.testing.assert_series_equal(
        transformed[
            "Total Fwd Packets"
        ],
        sample_frame[
            "Total Fwd Packets"
        ],
    )


def test_labels_and_targets_unchanged(
    config,
    plan,
    sample_frame,
):
    transformed, *_ = (
        transform_frame(
            sample_frame,
            config,
            plan,
        )
    )

    pd.testing.assert_series_equal(
        transformed["Label"],
        sample_frame["Label"],
    )

    pd.testing.assert_series_equal(
        transformed[
            "binary_target"
        ],
        sample_frame[
            "binary_target"
        ],
    )


def test_timing_features_use_same_scale(
    config,
    plan,
    sample_frame,
):
    (
        transformed,
        _,
        scales,
        _,
        _,
    ) = transform_frame(
        sample_frame,
        config,
        plan,
    )

    attack_a = 1

    assert scales.iloc[
        attack_a
    ] == pytest.approx(
        0.5
    )

    assert transformed.loc[
        attack_a,
        "Flow Duration",
    ] == pytest.approx(
        100.0
    )

    assert transformed.loc[
        attack_a,
        "Flow IAT Mean",
    ] == pytest.approx(
        10.0
    )


def test_rates_scale_inversely(
    config,
    plan,
    sample_frame,
):
    transformed, *_ = (
        transform_frame(
            sample_frame,
            config,
            plan,
        )
    )

    attack_a = 1

    assert transformed.loc[
        attack_a,
        "Flow Bytes/s",
    ] == pytest.approx(
        1000.0
    )

    assert transformed.loc[
        attack_a,
        "Flow Packets/s",
    ] == pytest.approx(
        100.0
    )


def test_duration_ceiling_constrains_scale(
    config,
    plan,
    sample_frame,
):
    frame = sample_frame.copy()

    frame.loc[
        2,
        "Flow Duration",
    ] = 800.0

    (
        transformed,
        sampled,
        applied,
        constrained,
        _,
    ) = transform_frame(
        frame,
        config,
        plan,
    )

    assert sampled.iloc[
        2
    ] == pytest.approx(
        2.0
    )

    assert applied.iloc[
        2
    ] == pytest.approx(
        1.5
    )

    assert constrained.iloc[
        2
    ]

    assert transformed.loc[
        2,
        "Flow Duration",
    ] == pytest.approx(
        1200.0
    )

    assert transformed.loc[
        2,
        "Flow IAT Mean",
    ] == pytest.approx(
        60.0
    )

    assert transformed.loc[
        2,
        "Flow Bytes/s",
    ] == pytest.approx(
        250.0 / 1.5
    )


def test_transformation_is_deterministic(
    config,
    sample_frame,
):
    random_plan = {
        "plan_version": 1,
        "strategy":
            "detector_independent_"
            "timing_morphing",
        "families": [
            {
                "label": "Attack A",
                "min_scale": 0.5,
                "max_scale": 0.8,
                "rationale":
                    "Synthetic test interval.",
            },
            {
                "label": "Attack B",
                "min_scale": 1.2,
                "max_scale": 1.8,
                "rationale":
                    "Synthetic test interval.",
            },
        ],
    }

    first = transform_frame(
        sample_frame,
        config,
        random_plan,
    )

    second = transform_frame(
        sample_frame,
        config,
        random_plan,
    )

    pd.testing.assert_frame_equal(
        first[0],
        second[0],
    )

    np.testing.assert_allclose(
        first[2].to_numpy(),
        second[2].to_numpy(),
    )


def test_integer_modifiable_columns_supported(
    config,
    plan,
    sample_frame,
):
    # This reproduces the real CIC schema
    # where some modifiable fields are stored
    # as integers.
    frame = sample_frame.copy()

    frame["Flow Duration"] = (
        frame[
            "Flow Duration"
        ].astype("int64")
    )

    frame["Flow Bytes/s"] = (
        frame[
            "Flow Bytes/s"
        ].astype("int64")
    )

    transformed, *_ = (
        transform_frame(
            frame,
            config,
            plan,
        )
    )

    assert (
        transformed[
            "Flow Duration"
        ].dtype
        == np.dtype(
            "float64"
        )
    )

    assert (
        transformed[
            "Flow Bytes/s"
        ].dtype
        == np.dtype(
            "float64"
        )
    )

    # Benign values remain numerically
    # unchanged despite the dtype promotion.
    assert transformed.loc[
        0,
        "Flow Duration",
    ] == pytest.approx(
        100.0
    )

    assert transformed.loc[
        0,
        "Flow Bytes/s",
    ] == pytest.approx(
        1000.0
    )

    # Attack A has a fixed scale of 0.5.
    assert transformed.loc[
        1,
        "Flow Duration",
    ] == pytest.approx(
        100.0
    )

    assert transformed.loc[
        1,
        "Flow Bytes/s",
    ] == pytest.approx(
        1000.0
    )

    # Non-modifiable fields should retain
    # their original values and dtype.
    pd.testing.assert_series_equal(
        transformed[
            "Destination Port"
        ],
        frame[
            "Destination Port"
        ],
    )