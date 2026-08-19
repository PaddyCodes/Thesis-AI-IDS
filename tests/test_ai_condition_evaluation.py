import pandas as pd
import pytest

from src.ai_condition_evaluation import (
    build_disagreement_rows,
    build_metric_comparison,
    ensure_not_already_run,
    load_experiment_config,
)


def test_experiment_config_loads():
    config = (
        load_experiment_config()
    )

    assert (
        config["experiment"]
        == "ai_assisted_held_out_test"
    )


def test_config_has_expected_dataset_hash():
    config = (
        load_experiment_config()
    )

    assert (
        config["dataset"]["sha256"]
        == (
            "8BA5578F5DDCD0A11120C51C"
            "9632E21A5077452D9E3F208D"
            "77BF2876D4A268AD"
        )
    )


def test_config_has_frozen_ai_plan_hash():
    config = (
        load_experiment_config()
    )

    assert (
        config[
            "ai_provenance"
        ][
            "plan_sha256"
        ]
        == (
            "A7A06D886B1D80F44DC18EAD"
            "8BF29E48B12AAAAD8D5442DFF"
            "64F9BC9D4D3FC20"
        )
    )


def test_config_has_scale_audit_hash():
    config = (
        load_experiment_config()
    )

    assert (
        config[
            "ai_provenance"
        ][
            "scale_audit_sha256"
        ]
        == (
            "2B9EE487BC81B1EC61CA41DFD"
            "CD40484A9531617D9C4872582"
            "A1CDF4500D7047"
        )
    )


def test_ensure_not_already_run_accepts_empty(
    tmp_path,
):
    output = (
        tmp_path
        / "result.csv"
    )

    ensure_not_already_run(
        [output]
    )


def test_ensure_not_already_run_rejects_existing(
    tmp_path,
):
    output = (
        tmp_path
        / "result.csv"
    )

    output.write_text(
        "existing",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="already exist",
    ):
        ensure_not_already_run(
            [output]
        )


def test_build_disagreement_rows():
    labels = [
        "BENIGN",
        "BENIGN",
        "Attack",
        "Attack",
    ]

    truth = [
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

    result = (
        build_disagreement_rows(
            labels,
            truth,
            rf,
            traditional,
        )
    )

    assert len(result) == 2

    assert result[
        "source_row"
    ].tolist() == [
        1,
        2,
    ]


def test_disagreement_correctness_flags():
    labels = [
        "BENIGN",
        "Attack",
    ]

    truth = [
        0,
        1,
    ]

    rf = [
        1,
        1,
    ]

    traditional = [
        0,
        0,
    ]

    result = (
        build_disagreement_rows(
            labels,
            truth,
            rf,
            traditional,
        )
    )

    assert (
        result[
            "random_forest_correct"
        ].tolist()
        == [
            False,
            True,
        ]
    )

    assert (
        result[
            "traditional_correct"
        ].tolist()
        == [
            True,
            False,
        ]
    )


def test_build_metric_comparison():
    original = {
        "random_forest": {
            "accuracy": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "f1": 0.9,
            "fpr": 0.1,
            "fnr": 0.1,
            "tn": 90,
            "fp": 10,
            "fn": 10,
            "tp": 90,
        },
        "traditional_rule_ids": {
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
            "fpr": 0.2,
            "fnr": 0.2,
            "tn": 80,
            "fp": 20,
            "fn": 20,
            "tp": 80,
        },
    }

    rf_ai = {
        "accuracy": 0.85,
        "precision": 0.85,
        "recall": 0.85,
        "f1": 0.85,
        "fpr": 0.15,
        "fnr": 0.15,
        "tn": 85,
        "fp": 15,
        "fn": 15,
        "tp": 85,
    }

    traditional_ai = {
        "accuracy": 0.75,
        "precision": 0.75,
        "recall": 0.75,
        "f1": 0.75,
        "fpr": 0.25,
        "fnr": 0.25,
        "tn": 75,
        "fp": 25,
        "fn": 25,
        "tp": 75,
    }

    result = (
        build_metric_comparison(
            original,
            rf_ai,
            traditional_ai,
        )
    )

    assert len(result) == 20

    row = result[
        (
            result["detector"]
            == "random_forest"
        )
        & (
            result["metric"]
            == "accuracy"
        )
    ].iloc[0]

    assert (
        row["original_value"]
        == pytest.approx(0.9)
    )

    assert (
        row["ai_assisted_value"]
        == pytest.approx(0.85)
    )

    assert (
        row["change"]
        == pytest.approx(-0.05)
    )


def test_disagreement_rejects_length_mismatch():
    with pytest.raises(
        ValueError,
        match="lengths do not match",
    ):
        build_disagreement_rows(
            ["BENIGN"],
            [0],
            [0, 1],
            [0],
        )