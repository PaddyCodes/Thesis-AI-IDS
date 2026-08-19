from pathlib import Path

import pandas as pd
import pytest

from src.final_evaluation import (
    agreement_summary,
    load_experiment_config,
    load_feature_names,
    sha256_file,
    validate_test_frame,
    verify_hash,
)


def test_experiment_config_loads():
    config = (
        load_experiment_config()
    )

    assert (
        config["experiment"]
        == "original_held_out_test"
    )


def test_config_has_expected_test_hash():
    config = (
        load_experiment_config()
    )

    assert (
        config["dataset"]["sha256"]
        == (
            "EB9118DA9D87345B0F13DA32"
            "B0581BE303463B99A7B6D735"
            "0C957623B612AD98"
        )
    )


def test_saved_feature_list():
    config = (
        load_experiment_config()
    )

    features = load_feature_names(
        config[
            "random_forest"
        ]["features_path"]
    )

    assert len(features) == 69
    assert len(features) == len(
        set(features)
    )


def test_sha256_file(tmp_path):
    file = (
        tmp_path
        / "test.txt"
    )

    file.write_text(
        "hello",
        encoding="utf-8",
    )

    assert sha256_file(file) == (
        "2CF24DBA5FB0A30E26E83B2AC5B9E29E"
        "1B161E5C1FA7425E73043362938B9824"
    )


def test_verify_hash_accepts_match(
    tmp_path,
):
    file = (
        tmp_path
        / "test.txt"
    )

    file.write_text(
        "hello",
        encoding="utf-8",
    )

    expected = sha256_file(
        file
    )

    result = verify_hash(
        file,
        expected,
        "Test file",
    )

    assert result == expected


def test_verify_hash_rejects_change(
    tmp_path,
):
    file = (
        tmp_path
        / "test.txt"
    )

    file.write_text(
        "hello",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="hash mismatch",
    ):
        verify_hash(
            file,
            "0" * 64,
            "Test file",
        )


def test_agreement_summary():
    y_true = [
        0,
        0,
        1,
        1,
    ]

    rf = [
        0,
        1,
        1,
        0,
    ]

    traditional = [
        0,
        0,
        0,
        0,
    ]

    result = agreement_summary(
        y_true,
        rf,
        traditional,
    )

    assert result[
        "records"
    ] == 4

    assert result[
        "both_correct"
    ] == 1

    assert result[
        "random_forest_only_correct"
    ] == 1

    assert result[
        "traditional_only_correct"
    ] == 1

    assert result[
        "both_wrong"
    ] == 1

    assert result[
        "prediction_disagreements"
    ] == 2


def test_validate_test_frame():
    config = {
        "dataset": {
            "expected_rows": 4,
            "expected_benign": 2,
            "expected_attacks": 2,
        }
    }

    df = pd.DataFrame(
        {
            "Label": [
                "BENIGN",
                "BENIGN",
                "Attack",
                "Attack",
            ],
            "binary_target": [
                0,
                0,
                1,
                1,
            ],
        }
    )

    validate_test_frame(
        df,
        config,
    )


def test_validate_test_frame_rejects_wrong_size():
    config = {
        "dataset": {
            "expected_rows": 5,
            "expected_benign": 2,
            "expected_attacks": 3,
        }
    }

    df = pd.DataFrame(
        {
            "Label": [
                "BENIGN",
                "Attack",
            ],
            "binary_target": [
                0,
                1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="row count",
    ):
        validate_test_frame(
            df,
            config,
        )