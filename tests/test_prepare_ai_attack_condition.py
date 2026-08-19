import json

import pandas as pd
import pytest

from src.prepare_ai_attack_condition import (
    build_ai_request,
    build_timing_summary,
    load_config,
    validate_source_frame,
)


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "Label": [
                "BENIGN",
                "BENIGN",
                "Attack A",
                "Attack A",
                "Attack B",
            ],
            "binary_target": [
                0,
                0,
                1,
                1,
                1,
            ],
            "Flow Duration": [
                100,
                200,
                20,
                40,
                1000,
            ],
        }
    )


def test_config_loads():
    config = load_config()

    assert (
        config["experiment"]
        == "ai_assisted_timing_morphing"
    )


def test_timing_summary_contains_groups(
    sample_frame,
):
    summary = build_timing_summary(
        sample_frame
    )

    groups = set(
        summary["group"]
    )

    assert groups == {
        "BENIGN",
        "Attack A",
        "Attack B",
    }


def test_timing_summary_counts(
    sample_frame,
):
    summary = build_timing_summary(
        sample_frame
    )

    counts = dict(
        zip(
            summary["group"],
            summary["count"],
        )
    )

    assert counts[
        "BENIGN"
    ] == 2

    assert counts[
        "Attack A"
    ] == 2

    assert counts[
        "Attack B"
    ] == 1


def test_ai_request_contains_all_attacks(
    sample_frame,
):
    config = load_config()

    summary = build_timing_summary(
        sample_frame
    )

    request = build_ai_request(
        config,
        summary,
        "A" * 64,
    )

    labels = {
        item["group"]
        for item
        in request[
            "attack_families"
        ]
    }

    assert labels == {
        "Attack A",
        "Attack B",
    }


def test_ai_request_excludes_detector_data(
    sample_frame,
):
    config = load_config()

    summary = build_timing_summary(
        sample_frame
    )

    request = build_ai_request(
        config,
        summary,
        "A" * 64,
    )

    text = json.dumps(
        request
    ).lower()

    assert (
        request[
            "detector_information_provided"
        ]
        is False
    )

    assert (
        "random_forest"
        not in text
    )

    assert (
        "traditional_ids"
        not in text
    )


def test_validate_source_frame():
    config = {
        "source_test": {
            "expected_rows": 4,
            "expected_benign": 2,
            "expected_attacks": 2,
        }
    }

    df = pd.DataFrame(
        {
            "binary_target": [
                0,
                0,
                1,
                1,
            ]
        }
    )

    validate_source_frame(
        df,
        config,
    )


def test_validate_source_frame_rejects_change():
    config = {
        "source_test": {
            "expected_rows": 4,
            "expected_benign": 2,
            "expected_attacks": 2,
        }
    }

    df = pd.DataFrame(
        {
            "binary_target": [
                0,
                1,
                1,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="row count",
    ):
        validate_source_frame(
            df,
            config,
        )