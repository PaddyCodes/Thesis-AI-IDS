from pathlib import Path

import pandas as pd
import pytest


SPLIT_DIR = Path("data/processed/splits")

TRAIN_FILE = SPLIT_DIR / "train.csv"
VALIDATION_FILE = SPLIT_DIR / "validation.csv"
TEST_FILE = SPLIT_DIR / "test.csv"


@pytest.fixture(scope="session")
def splits():
    # Load these once as they're fairly big files
    for file in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]:
        assert file.exists(), (
            f"{file} not found. "
            "Run 'python -m src.split_data' first."
        )

    return {
        "train": pd.read_csv(TRAIN_FILE),
        "validation": pd.read_csv(VALIDATION_FILE),
        "test": pd.read_csv(TEST_FILE),
    }


def test_expected_split_sizes(splits):
    assert len(splits["train"]) == 1_764_558
    assert len(splits["validation"]) == 378_120
    assert len(splits["test"]) == 378_120


def test_all_records_are_accounted_for(splits):
    total = (
        len(splits["train"])
        + len(splits["validation"])
        + len(splits["test"])
    )

    assert total == 2_520_798


def test_split_percentages(splits):
    total = 2_520_798

    train_ratio = len(splits["train"]) / total
    validation_ratio = len(splits["validation"]) / total
    test_ratio = len(splits["test"]) / total

    assert abs(train_ratio - 0.70) < 0.00001
    assert abs(validation_ratio - 0.15) < 0.00001
    assert abs(test_ratio - 0.15) < 0.00001


def test_both_classes_exist_in_each_split(splits):
    for name, df in splits.items():
        classes = set(
            df["binary_target"].unique()
        )

        assert classes == {0, 1}, (
            f"Unexpected classes in {name}"
        )


def test_class_balance_is_preserved(splits):
    expected_attack_ratio = (
        425_741 / 2_520_798
    )

    for name, df in splits.items():
        attack_ratio = (
            df["binary_target"].mean()
        )

        assert abs(
            attack_ratio - expected_attack_ratio
        ) < 0.0001, (
            f"Class balance changed in {name}"
        )


def test_labels_match_binary_target(splits):
    for name, df in splits.items():
        expected = (
            df["Label"] != "BENIGN"
        ).astype(int)

        assert (
            df["binary_target"] == expected
        ).all(), (
            f"Target mismatch found in {name}"
        )


def test_same_features_in_every_split(splits):
    train_columns = list(
        splits["train"].columns
    )

    assert list(
        splits["validation"].columns
    ) == train_columns

    assert list(
        splits["test"].columns
    ) == train_columns


def test_all_attack_types_are_present(splits):
    """
    Check the rare attack types haven't disappeared
    from one of the partitions.
    """

    expected_labels = {
        "BENIGN",
        "DoS Hulk",
        "DDoS",
        "PortScan",
        "DoS GoldenEye",
        "FTP-Patator",
        "DoS slowloris",
        "DoS Slowhttptest",
        "SSH-Patator",
        "Bot",
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Infiltration",
        "Web Attack - SQL Injection",
        "Heartbleed",
    }

    for name, df in splits.items():
        labels = set(
            df["Label"].unique()
        )

        missing = (
            expected_labels - labels
        )

        assert not missing, (
            f"{name} is missing labels: {missing}"
        )


def test_no_missing_values_in_splits(splits):
    for name, df in splits.items():
        assert not df.isna().any().any(), (
            f"Missing values found in {name}"
        )