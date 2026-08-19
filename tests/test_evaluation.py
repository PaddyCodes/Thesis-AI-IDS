import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    attack_family_recall,
    calculate_metrics,
)


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 1])

    metrics = calculate_metrics(
        y_true,
        predictions,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0

    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2


def test_known_confusion_matrix():
    y_true = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
    ])

    predictions = np.array([
        0, 0, 0, 1,
        1, 1, 0, 0,
    ])

    metrics = calculate_metrics(
        y_true,
        predictions,
    )

    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 2
    assert metrics["tp"] == 2

    assert metrics["accuracy"] == pytest.approx(
        0.625
    )

    assert metrics["precision"] == pytest.approx(
        2 / 3
    )

    assert metrics["recall"] == pytest.approx(
        0.5
    )

    assert metrics["f1"] == pytest.approx(
        4 / 7
    )

    assert metrics["fpr"] == pytest.approx(
        0.25
    )

    assert metrics["fnr"] == pytest.approx(
        0.5
    )


def test_all_attacks_missed():
    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 0, 0])

    metrics = calculate_metrics(
        y_true,
        predictions,
    )

    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["fnr"] == 1.0

    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 2
    assert metrics["tp"] == 0


def test_all_benign_records_flagged():
    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([1, 1, 1, 1])

    metrics = calculate_metrics(
        y_true,
        predictions,
    )

    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 1.0
    assert metrics["fnr"] == 0.0

    assert metrics["tn"] == 0
    assert metrics["fp"] == 2
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2


def test_attack_family_recall():
    labels = pd.Series([
        "BENIGN",
        "BENIGN",
        "DDoS",
        "DDoS",
        "Bot",
        "Bot",
    ])

    y_true = np.array([
        0,
        0,
        1,
        1,
        1,
        1,
    ])

    predictions = np.array([
        0,
        0,
        1,
        0,
        1,
        1,
    ])

    results = attack_family_recall(
        labels,
        y_true,
        predictions,
    )

    by_attack = {
        result["attack"]: result
        for result in results
    }

    assert "BENIGN" not in by_attack

    assert by_attack["DDoS"]["samples"] == 2
    assert by_attack["DDoS"]["detected"] == 1
    assert by_attack["DDoS"]["missed"] == 1
    assert by_attack["DDoS"]["recall"] == 0.5

    assert by_attack["Bot"]["samples"] == 2
    assert by_attack["Bot"]["detected"] == 2
    assert by_attack["Bot"]["missed"] == 0
    assert by_attack["Bot"]["recall"] == 1.0


def test_attack_family_with_no_detections():
    labels = pd.Series([
        "BENIGN",
        "Bot",
        "Bot",
    ])

    y_true = np.array([
        0,
        1,
        1,
    ])

    predictions = np.array([
        0,
        0,
        0,
    ])

    results = attack_family_recall(
        labels,
        y_true,
        predictions,
    )

    assert len(results) == 1

    bot = results[0]

    assert bot["attack"] == "Bot"
    assert bot["samples"] == 2
    assert bot["detected"] == 0
    assert bot["missed"] == 2
    assert bot["recall"] == 0.0