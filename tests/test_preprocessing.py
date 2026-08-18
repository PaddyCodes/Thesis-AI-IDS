from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CONSTANT_FEATURES,
    DUPLICATE_FEATURES,
)


PROCESSED_DATASET = Path(
    "data/processed/cicids2017_clean.csv"
)


@pytest.fixture(scope="session")
def processed_df():
    """
    Load the processed CIC-IDS2017 dataset once
    for the complete pytest session.

    Session scope avoids repeatedly loading the
    ~2.5 million row dataset for every individual test.
    """

    assert PROCESSED_DATASET.exists(), (
        "Processed dataset does not exist. "
        "Run 'python -m src.preprocessing' first."
    )

    return pd.read_csv(
        PROCESSED_DATASET
    )


def test_expected_dataset_dimensions(
    processed_df,
):
    """
    Confirm that the deterministic preprocessing
    pipeline produces the expected dataset size.
    """

    assert len(processed_df) == 2_520_798

    assert len(
        processed_df.columns
    ) == 71


def test_no_missing_values(
    processed_df,
):
    """
    Processed data must contain no NaN values.
    """

    total_missing = int(
        processed_df
        .isna()
        .sum()
        .sum()
    )

    assert total_missing == 0


def test_no_infinite_values(
    processed_df,
):
    """
    Numerical model data must contain
    no positive or negative infinity.
    """

    numeric_df = (
        processed_df
        .select_dtypes(
            include=np.number
        )
    )

    total_infinite = int(
        np.isinf(
            numeric_df.to_numpy()
        ).sum()
    )

    assert total_infinite == 0


def test_flow_rate_features_are_valid(
    processed_df,
):
    """
    Explicitly validate the CIC-IDS2017
    flow-rate fields known to contain
    invalid values in the raw dataset.
    """

    rate_features = [
        "Flow Bytes/s",
        "Flow Packets/s",
    ]

    for feature in rate_features:

        assert (
            feature
            in processed_df.columns
        )

        assert (
            processed_df[
                feature
            ]
            .isna()
            .sum()
            == 0
        )

        assert (
            np.isinf(
                processed_df[
                    feature
                ].to_numpy()
            ).sum()
            == 0
        )


def test_no_duplicate_rows(
    processed_df,
):
    """
    No exact duplicate records should remain.
    """

    duplicates = int(
        processed_df
        .duplicated()
        .sum()
    )

    assert duplicates == 0


def test_binary_target_is_valid(
    processed_df,
):
    """
    Binary IDS target must contain exactly
    benign=0 and attack=1.
    """

    target_values = set(
        processed_df[
            "binary_target"
        ].unique()
    )

    assert target_values == {0, 1}


def test_binary_target_matches_labels(
    processed_df,
):
    """
    Verify that BENIGN records map to 0
    and all attack labels map to 1.
    """

    expected_target = (
        processed_df["Label"]
        != "BENIGN"
    ).astype(int)

    assert (
        processed_df[
            "binary_target"
        ]
        == expected_target
    ).all()


def test_no_column_whitespace(
    processed_df,
):
    """
    Column names must not contain leading
    or trailing whitespace.
    """

    assert all(
        column == column.strip()
        for column
        in processed_df.columns
    )


def test_no_duplicate_column_names(
    processed_df,
):
    """
    Processed dataset must not contain
    duplicate column names.
    """

    assert not (
        processed_df
        .columns
        .duplicated()
        .any()
    )


def test_no_corrupted_labels(
    processed_df,
):
    """
    Known Unicode replacement characters
    must not remain in attack labels.
    """

    corrupted = (
        processed_df[
            "Label"
        ]
        .str.contains(
            "\ufffd",
            regex=False,
            na=False,
        )
        .any()
    )

    assert not corrupted


def test_no_label_whitespace(
    processed_df,
):
    """
    Attack labels must not contain leading
    or trailing whitespace.
    """

    assert (
        processed_df[
            "Label"
        ]
        == processed_df[
            "Label"
        ].str.strip()
    ).all()


def test_known_web_attack_labels_fixed(
    processed_df,
):
    """
    Verify that known corrupted CIC-IDS2017
    web attack labels have been normalised.
    """

    labels = set(
        processed_df[
            "Label"
        ].unique()
    )

    expected_labels = {
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - SQL Injection",
    }

    assert expected_labels.issubset(
        labels
    )


def test_all_model_features_are_numeric(
    processed_df,
):
    """
    Every predictor supplied to the
    machine-learning model must be numeric.
    """

    feature_columns = [
        column
        for column
        in processed_df.columns
        if column not in {
            "Label",
            "binary_target",
        }
    ]

    assert all(
        pd.api.types.is_numeric_dtype(
            processed_df[column]
        )
        for column
        in feature_columns
    )


def test_expected_feature_count(
    processed_df,
):
    """
    Structural preprocessing should leave
    exactly 69 numeric predictor features.
    """

    feature_columns = [
        column
        for column
        in processed_df.columns
        if column not in {
            "Label",
            "binary_target",
        }
    ]

    assert len(
        feature_columns
    ) == 69


def test_no_constant_features(
    processed_df,
):
    """
    No predictor should contain only
    one unique value.
    """

    feature_columns = [
        column
        for column
        in processed_df.columns
        if column not in {
            "Label",
            "binary_target",
        }
    ]

    unique_counts = (
        processed_df[
            feature_columns
        ]
        .nunique(
            dropna=False
        )
    )

    constant_features = (
        unique_counts[
            unique_counts <= 1
        ]
        .index
        .tolist()
    )

    assert constant_features == []


def test_known_redundant_features_removed(
    processed_df,
):
    """
    Constant and duplicated features identified
    during exploratory validation must not remain
    in the processed dataset.
    """

    known_redundant = (
        CONSTANT_FEATURES
        + DUPLICATE_FEATURES
    )

    remaining = [
        feature
        for feature
        in known_redundant
        if feature
        in processed_df.columns
    ]

    assert remaining == []


def test_class_distribution_contains_both_classes(
    processed_df,
):
    """
    Both benign and malicious records must remain
    after preprocessing.
    """

    counts = (
        processed_df[
            "binary_target"
        ]
        .value_counts()
    )

    assert 0 in counts.index
    assert 1 in counts.index

    assert counts[0] > 0
    assert counts[1] > 0