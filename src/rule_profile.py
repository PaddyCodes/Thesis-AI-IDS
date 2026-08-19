from pathlib import Path

import pandas as pd


TRAIN_FILE = Path(
    "data/processed/splits/train.csv"
)

METRICS_DIR = Path(
    "outputs/metrics"
)

# These are features that make sense for a fixed flow-based IDS.
# They are chosen from their network meaning, not from the ML model.
PROFILE_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "SYN Flag Count",
    "RST Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "FIN Flag Count",
    "Average Packet Size",
    "Down/Up Ratio",
    "Packet Length Mean",
    "Packet Length Std",
]

QUANTILES = [
    0.001,
    0.01,
    0.05,
    0.50,
    0.95,
    0.99,
    0.999,
]


def load_training_data():
    # Only the training partition is allowed here.
    # Test data must stay untouched.
    columns = [
        "Label",
        "binary_target",
        "Destination Port",
        *PROFILE_FEATURES,
    ]

    print("Loading training data...")

    df = pd.read_csv(
        TRAIN_FILE,
        usecols=columns,
    )

    print(
        f"Training records: {len(df):,}"
    )

    return df


def profile_group(
    df,
    scope,
    group_name,
):
    rows = []

    for feature in PROFILE_FEATURES:
        values = df[feature]

        quantiles = values.quantile(
            QUANTILES
        )

        rows.append(
            {
                "scope": scope,
                "group": group_name,
                "feature": feature,
                "count": int(
                    values.count()
                ),
                "min": float(
                    values.min()
                ),
                "q001": float(
                    quantiles.loc[0.001]
                ),
                "q01": float(
                    quantiles.loc[0.01]
                ),
                "q05": float(
                    quantiles.loc[0.05]
                ),
                "median": float(
                    quantiles.loc[0.50]
                ),
                "q95": float(
                    quantiles.loc[0.95]
                ),
                "q99": float(
                    quantiles.loc[0.99]
                ),
                "q999": float(
                    quantiles.loc[0.999]
                ),
                "max": float(
                    values.max()
                ),
            }
        )

    return rows


def build_feature_profile(df):
    rows = []

    benign_df = df[
        df["binary_target"] == 0
    ]

    attack_df = df[
        df["binary_target"] == 1
    ]

    rows.extend(
        profile_group(
            benign_df,
            "binary",
            "BENIGN",
        )
    )

    rows.extend(
        profile_group(
            attack_df,
            "binary",
            "ATTACK",
        )
    )

    # Keep attack families separate as fixed signatures may
    # behave very differently across different attacks.
    attack_labels = sorted(
        label
        for label in df["Label"].unique()
        if label != "BENIGN"
    )

    for label in attack_labels:
        attack_family_df = df[
            df["Label"] == label
        ]

        rows.extend(
            profile_group(
                attack_family_df,
                "attack_family",
                label,
            )
        )

    return pd.DataFrame(rows)


def profile_ports_for_group(
    df,
    scope,
    group_name,
    limit=15,
):
    counts = (
        df["Destination Port"]
        .value_counts()
        .head(limit)
    )

    total = len(df)

    rows = []

    for port, count in counts.items():
        rows.append(
            {
                "scope": scope,
                "group": group_name,
                "destination_port": int(port),
                "count": int(count),
                "percentage": float(
                    count / total * 100
                ),
            }
        )

    return rows


def build_port_profile(df):
    rows = []

    benign_df = df[
        df["binary_target"] == 0
    ]

    attack_df = df[
        df["binary_target"] == 1
    ]

    rows.extend(
        profile_ports_for_group(
            benign_df,
            "binary",
            "BENIGN",
        )
    )

    rows.extend(
        profile_ports_for_group(
            attack_df,
            "binary",
            "ATTACK",
        )
    )

    attack_labels = sorted(
        label
        for label in df["Label"].unique()
        if label != "BENIGN"
    )

    for label in attack_labels:
        attack_family_df = df[
            df["Label"] == label
        ]

        rows.extend(
            profile_ports_for_group(
                attack_family_df,
                "attack_family",
                label,
            )
        )

    return pd.DataFrame(rows)


def build_rule_comparison(profile_df):
    benign = profile_df[
        (profile_df["scope"] == "binary")
        & (profile_df["group"] == "BENIGN")
    ][
        [
            "feature",
            "q001",
            "q01",
            "q99",
            "q999",
        ]
    ].copy()

    benign = benign.rename(
        columns={
            "q001": "benign_q001",
            "q01": "benign_q01",
            "q99": "benign_q99",
            "q999": "benign_q999",
        }
    )

    attack = profile_df[
        (profile_df["scope"] == "binary")
        & (profile_df["group"] == "ATTACK")
    ][
        [
            "feature",
            "median",
            "q95",
            "q99",
        ]
    ].copy()

    attack = attack.rename(
        columns={
            "median": "attack_median",
            "q95": "attack_q95",
            "q99": "attack_q99",
        }
    )

    return benign.merge(
        attack,
        on="feature",
        how="inner",
    )


def main():
    print("=" * 70)
    print("TRADITIONAL IDS - TRAINING PROFILE")
    print("=" * 70)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training split not found: "
            f"{TRAIN_FILE.resolve()}"
        )

    df = load_training_data()

    print("\nClass distribution:")

    print(
        df["binary_target"]
        .value_counts()
        .sort_index()
    )

    print("\nBuilding feature profile...")

    feature_profile = (
        build_feature_profile(df)
    )

    print("Building port profile...")

    port_profile = (
        build_port_profile(df)
    )

    rule_comparison = (
        build_rule_comparison(
            feature_profile
        )
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_output = (
        METRICS_DIR
        / "traditional_ids_training_profile.csv"
    )

    port_output = (
        METRICS_DIR
        / "traditional_ids_port_profile.csv"
    )

    comparison_output = (
        METRICS_DIR
        / "traditional_ids_rule_comparison.csv"
    )

    feature_profile.to_csv(
        feature_output,
        index=False,
    )

    port_profile.to_csv(
        port_output,
        index=False,
    )

    rule_comparison.to_csv(
        comparison_output,
        index=False,
    )

    print("\n" + "=" * 70)
    print("RULE THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        rule_comparison.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(feature_output)
    print(port_output)
    print(comparison_output)

    print(
        "\nNo validation or test data "
        "was used."
    )


if __name__ == "__main__":
    main()