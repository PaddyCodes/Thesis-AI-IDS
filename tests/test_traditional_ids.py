import pandas as pd
import pytest

from src.traditional_ids import (
    apply_rules,
    condition_mask,
    load_rules,
    required_features,
)


@pytest.fixture(scope="session")
def rule_config():
    return load_rules()


@pytest.fixture(scope="session")
def rules(rule_config):
    return rule_config["rules"]


def base_flow(rules):
    # Start with a flow that should not match
    # any of the configured signatures.
    row = {
        feature: 0
        for feature
        in required_features(rules)
    }

    row.update(
        {
            "Destination Port": 443,
            "Flow Duration": 10000,
            "Total Fwd Packets": 10,
            "Total Backward Packets": 10,
            "Average Packet Size": 100,
            "Packet Length Std": 100,
            "Down/Up Ratio": 1,
            "SYN Flag Count": 0,
            "ACK Flag Count": 0,
        }
    )

    return row


def test_rule_config_loads(
    rule_config,
):
    assert rule_config[
        "version"
    ] == 1

    assert len(
        rule_config["rules"]
    ) == 8


def test_rule_names_are_unique(
    rules,
):
    names = [
        rule["name"]
        for rule in rules
    ]

    assert len(names) == len(
        set(names)
    )


def test_benign_flow_does_not_match(
    rules,
):
    df = pd.DataFrame(
        [
            base_flow(rules)
        ]
    )

    predictions, reasons, _ = (
        apply_rules(
            df,
            rules,
        )
    )

    assert predictions.tolist() == [0]
    assert reasons.iloc[0] == ""


def test_portscan_rule(
    rules,
):
    row = base_flow(rules)

    row.update(
        {
            "Flow Duration": 50,
            "Total Fwd Packets": 1,
            "Total Backward Packets": 1,
            "Average Packet Size": 3,
            "Packet Length Std": 2.3,
        }
    )

    predictions, reasons, _ = (
        apply_rules(
            pd.DataFrame([row]),
            rules,
        )
    )

    assert predictions.tolist() == [1]

    assert (
        reasons.iloc[0]
        == "PORTSCAN_SHORT_FLOW"
    )


def test_ftp_brute_force_rule(
    rules,
):
    row = base_flow(rules)

    row.update(
        {
            "Destination Port": 21,
            "Flow Duration": 8000000,
            "Total Fwd Packets": 9,
            "Total Backward Packets": 15,
            "Average Packet Size": 12,
            "Packet Length Std": 13,
            "Down/Up Ratio": 1,
        }
    )

    predictions, reasons, _ = (
        apply_rules(
            pd.DataFrame([row]),
            rules,
        )
    )

    assert predictions.tolist() == [1]

    assert (
        reasons.iloc[0]
        == "FTP_BRUTE_FORCE"
    )


def test_ssh_brute_force_rule(
    rules,
):
    row = base_flow(rules)

    row.update(
        {
            "Destination Port": 22,
            "Flow Duration": 12000000,
            "Total Fwd Packets": 21,
            "Total Backward Packets": 32,
            "Average Packet Size": 90,
            "Packet Length Std": 190,
            "Down/Up Ratio": 1,
        }
    )

    predictions, reasons, _ = (
        apply_rules(
            pd.DataFrame([row]),
            rules,
        )
    )

    assert predictions.tolist() == [1]

    assert (
        reasons.iloc[0]
        == "SSH_BRUTE_FORCE"
    )


def test_bot_rule(
    rules,
):
    row = base_flow(rules)

    row.update(
        {
            "Destination Port": 8080,
            "Flow Duration": 70000,
            "Total Fwd Packets": 3,
            "Total Backward Packets": 3,
            "Average Packet Size": 9,
            "Packet Length Std": 4,
        }
    )

    predictions, reasons, _ = (
        apply_rules(
            pd.DataFrame([row]),
            rules,
        )
    )

    assert predictions.tolist() == [1]

    assert (
        reasons.iloc[0]
        == "BOT_8080"
    )


def test_http_flood_rule(
    rules,
):
    row = base_flow(rules)

    row.update(
        {
            "Destination Port": 80,
            "Flow Duration": 2000000,
            "Total Fwd Packets": 6,
            "Total Backward Packets": 5,
            "Average Packet Size": 900,
            "Packet Length Std": 1800,
        }
    )

    predictions, reasons, _ = (
        apply_rules(
            pd.DataFrame([row]),
            rules,
        )
    )

    assert predictions.tolist() == [1]

    assert (
        reasons.iloc[0]
        == "HTTP_FLOOD"
    )


def test_slow_http_rule(
    rules,
):
    row = base_flow(rules)

    row.update(
        {
            "Destination Port": 80,
            "Flow Duration": 60000000,
            "Total Fwd Packets": 7,
            "Total Backward Packets": 1,
            "Average Packet Size": 100,
            "Packet Length Std": 100,
        }
    )

    predictions, reasons, _ = (
        apply_rules(
            pd.DataFrame([row]),
            rules,
        )
    )

    assert predictions.tolist() == [1]

    assert (
        reasons.iloc[0]
        == "SLOW_HTTP_DOS"
    )


def test_heartbleed_rule(
    rules,
):
    row = base_flow(rules)

    row.update(
        {
            "Destination Port": 444,
            "Flow Duration": 119000000,
            "Total Fwd Packets": 2800,
            "Total Backward Packets": 2050,
            "Average Packet Size": 1620,
            "Packet Length Std": 2450,
        }
    )

    predictions, reasons, _ = (
        apply_rules(
            pd.DataFrame([row]),
            rules,
        )
    )

    assert predictions.tolist() == [1]

    assert (
        reasons.iloc[0]
        == "HEARTBLEED_LARGE_FLOW"
    )


def test_infiltration_rule(
    rules,
):
    row = base_flow(rules)

    row.update(
        {
            "Destination Port": 444,
            "Flow Duration": 80000000,
            "Total Fwd Packets": 25,
            "Total Backward Packets": 25,
            "Average Packet Size": 140,
            "Packet Length Std": 280,
            "SYN Flag Count": 1,
            "ACK Flag Count": 1,
        }
    )

    predictions, reasons, _ = (
        apply_rules(
            pd.DataFrame([row]),
            rules,
        )
    )

    assert predictions.tolist() == [1]

    assert (
        reasons.iloc[0]
        == "INFILTRATION_444"
    )


def test_missing_feature_is_rejected(
    rules,
):
    condition = {
        "feature": "Missing Feature",
        "operator": "ge",
        "value": 1,
    }

    df = pd.DataFrame(
        {
            "Other Feature": [1]
        }
    )

    with pytest.raises(
        ValueError,
        match="Required feature",
    ):
        condition_mask(
            df,
            condition,
        )


def test_predictions_are_binary(
    rules,
):
    benign = base_flow(rules)

    attack = base_flow(rules)

    attack.update(
        {
            "Flow Duration": 50,
            "Total Fwd Packets": 1,
            "Total Backward Packets": 1,
            "Average Packet Size": 3,
            "Packet Length Std": 2,
        }
    )

    predictions, _, _ = apply_rules(
        pd.DataFrame(
            [
                benign,
                attack,
            ]
        ),
        rules,
    )

    assert set(
        predictions
    ).issubset(
        {0, 1}
    )