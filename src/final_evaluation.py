from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd

from src.evaluation import (
    attack_family_recall,
    calculate_metrics,
    print_attack_family_results,
    print_metrics,
)
from src.traditional_ids import (
    apply_rules,
    load_rules,
    required_features,
)


CONFIG_FILE = Path(
    "configs/final_experiment.json"
)

OUTPUT_DIR = Path(
    "outputs/metrics"
)

METADATA_FILE = OUTPUT_DIR / (
    "original_test_metadata.json"
)


def sha256_file(path):
    hasher = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(65536),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest().upper()


def verify_hash(
    path,
    expected_hash,
    name,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: "
            f"{path.resolve()}"
        )

    actual_hash = sha256_file(path)

    if (
        actual_hash.upper()
        != expected_hash.upper()
    ):
        raise ValueError(
            f"{name} hash mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    return actual_hash


def load_experiment_config(
    path=CONFIG_FILE,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Experiment config not found: "
            f"{path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    required_sections = {
        "experiment",
        "dataset",
        "random_forest",
        "traditional_ids",
    }

    missing = (
        required_sections
        - set(config)
    )

    if missing:
        raise ValueError(
            "Experiment config is missing: "
            + ", ".join(sorted(missing))
        )

    return config


def load_feature_names(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature list not found: "
            f"{path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        features = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not features:
        raise ValueError(
            "Feature list is empty."
        )

    if len(features) != len(
        set(features)
    ):
        raise ValueError(
            "Feature list contains duplicates."
        )

    return features


def verify_frozen_inputs(config):
    dataset = config["dataset"]
    model = config["random_forest"]
    traditional = config[
        "traditional_ids"
    ]

    print(
        "Verifying frozen experiment "
        "artifacts..."
    )

    test_hash = verify_hash(
        dataset["path"],
        dataset["sha256"],
        "Held-out test set",
    )

    model_hash = verify_hash(
        model["model_path"],
        model["model_sha256"],
        "Random Forest model",
    )

    features_hash = verify_hash(
        model["features_path"],
        model["features_sha256"],
        "Random Forest feature list",
    )

    rules_hash = verify_hash(
        traditional["rules_path"],
        traditional["rules_sha256"],
        "Traditional IDS rules",
    )

    features = load_feature_names(
        model["features_path"]
    )

    expected_count = model[
        "expected_feature_count"
    ]

    if len(features) != expected_count:
        raise ValueError(
            "Unexpected model feature count. "
            f"Expected {expected_count}, "
            f"found {len(features)}."
        )

    print(
        "All frozen artifact hashes "
        "verified."
    )

    return {
        "test_sha256": test_hash,
        "model_sha256": model_hash,
        "features_sha256":
            features_hash,
        "rules_sha256": rules_hash,
        "model_features": features,
    }


def validate_test_frame(
    df,
    config,
):
    dataset = config["dataset"]

    expected_rows = dataset[
        "expected_rows"
    ]

    if len(df) != expected_rows:
        raise ValueError(
            "Unexpected test row count. "
            f"Expected {expected_rows:,}, "
            f"found {len(df):,}."
        )

    counts = (
        df["binary_target"]
        .value_counts()
        .to_dict()
    )

    benign = int(
        counts.get(0, 0)
    )

    attacks = int(
        counts.get(1, 0)
    )

    if (
        benign
        != dataset["expected_benign"]
    ):
        raise ValueError(
            "Unexpected benign count. "
            f"Expected "
            f"{dataset['expected_benign']:,}, "
            f"found {benign:,}."
        )

    if (
        attacks
        != dataset["expected_attacks"]
    ):
        raise ValueError(
            "Unexpected attack count. "
            f"Expected "
            f"{dataset['expected_attacks']:,}, "
            f"found {attacks:,}."
        )

    expected_targets = {
        0,
        1,
    }

    if (
        set(
            df["binary_target"]
            .unique()
        )
        != expected_targets
    ):
        raise ValueError(
            "Test target is not binary."
        )


def load_test_data(
    config,
    model_features,
    rule_features,
):
    test_path = Path(
        config["dataset"]["path"]
    )

    required_columns = list(
        dict.fromkeys(
            [
                "Label",
                "binary_target",
                *model_features,
                *rule_features,
            ]
        )
    )

    print(
        "\nLoading held-out test data..."
    )

    df = pd.read_csv(
        test_path,
        usecols=required_columns,
    )

    validate_test_frame(
        df,
        config,
    )

    print(
        f"Test records: {len(df):,}"
    )

    print(
        "Test partition integrity "
        "checks passed."
    )

    return df


def evaluate_predictions(
    labels,
    y_true,
    predictions,
):
    metrics = calculate_metrics(
        y_true,
        predictions,
    )

    family_results = (
        attack_family_recall(
            labels,
            y_true,
            predictions,
        )
    )

    return (
        metrics,
        family_results,
    )


def agreement_summary(
    y_true,
    rf_predictions,
    traditional_predictions,
):
    truth = pd.Series(
        y_true
    ).reset_index(drop=True)

    rf = pd.Series(
        rf_predictions
    ).reset_index(drop=True)

    traditional = pd.Series(
        traditional_predictions
    ).reset_index(drop=True)

    if not (
        len(truth)
        == len(rf)
        == len(traditional)
    ):
        raise ValueError(
            "Prediction lengths do not match."
        )

    rf_correct = rf == truth

    traditional_correct = (
        traditional == truth
    )

    return {
        "records": int(len(truth)),
        "both_correct": int(
            (
                rf_correct
                & traditional_correct
            ).sum()
        ),
        "random_forest_only_correct":
            int(
                (
                    rf_correct
                    & ~traditional_correct
                ).sum()
            ),
        "traditional_only_correct":
            int(
                (
                    ~rf_correct
                    & traditional_correct
                ).sum()
            ),
        "both_wrong": int(
            (
                ~rf_correct
                & ~traditional_correct
            ).sum()
        ),
        "prediction_disagreements":
            int(
                (
                    rf
                    != traditional
                ).sum()
            ),
    }


def serialise_metrics(metrics):
    integer_metrics = {
        "tn",
        "fp",
        "fn",
        "tp",
    }

    return {
        key: (
            int(value)
            if key in integer_metrics
            else float(value)
        )
        for key, value
        in metrics.items()
    }


def save_results(
    config,
    verified,
    rf_metrics,
    rf_family,
    traditional_metrics,
    traditional_family,
    rule_matches,
    agreement,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rf_row = {
        "detector": "random_forest",
        **rf_metrics,
    }

    traditional_row = {
        "detector":
            "traditional_rule_ids",
        **traditional_metrics,
    }

    pd.DataFrame(
        [rf_row]
    ).to_csv(
        OUTPUT_DIR
        / "original_test_random_forest_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        [traditional_row]
    ).to_csv(
        OUTPUT_DIR
        / "original_test_traditional_ids_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            rf_row,
            traditional_row,
        ]
    ).to_csv(
        OUTPUT_DIR
        / "original_test_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        rf_family
    ).to_csv(
        OUTPUT_DIR
        / "original_test_random_forest_attack_recall.csv",
        index=False,
    )

    pd.DataFrame(
        traditional_family
    ).to_csv(
        OUTPUT_DIR
        / "original_test_traditional_ids_attack_recall.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "rule": rule,
                "matches": count,
            }
            for rule, count
            in rule_matches.items()
        ]
    ).to_csv(
        OUTPUT_DIR
        / "original_test_traditional_rule_matches.csv",
        index=False,
    )

    pd.DataFrame(
        [agreement]
    ).to_csv(
        OUTPUT_DIR
        / "original_test_agreement.csv",
        index=False,
    )

    metadata = {
        "experiment":
            config["experiment"],
        "executed_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "dataset": {
            "path":
                config["dataset"]["path"],
            "sha256":
                verified["test_sha256"],
            "records":
                config["dataset"][
                    "expected_rows"
                ],
            "benign":
                config["dataset"][
                    "expected_benign"
                ],
            "attacks":
                config["dataset"][
                    "expected_attacks"
                ],
        },
        "random_forest": {
            "model_sha256":
                verified[
                    "model_sha256"
                ],
            "features_sha256":
                verified[
                    "features_sha256"
                ],
            "feature_count":
                len(
                    verified[
                        "model_features"
                    ]
                ),
            "metrics":
                serialise_metrics(
                    rf_metrics
                ),
        },
        "traditional_ids": {
            "rules_sha256":
                verified[
                    "rules_sha256"
                ],
            "metrics":
                serialise_metrics(
                    traditional_metrics
                ),
            "rule_matches":
                {
                    key: int(value)
                    for key, value
                    in rule_matches.items()
                },
        },
        "agreement": agreement,
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        "\nOriginal held-out test "
        "results saved."
    )


def ensure_not_already_run():
    if METADATA_FILE.exists():
        raise RuntimeError(
            "Original held-out test results "
            "already exist. Refusing to "
            "overwrite them."
        )


def main():
    print("=" * 70)
    print("FINAL HELD-OUT TEST EVALUATION")
    print("=" * 70)

    ensure_not_already_run()

    config = (
        load_experiment_config()
    )

    verified = (
        verify_frozen_inputs(
            config
        )
    )

    model_features = verified[
        "model_features"
    ]

    rule_config = load_rules(
        Path(
            config[
                "traditional_ids"
            ]["rules_path"]
        )
    )

    rules = rule_config[
        "rules"
    ]

    rule_features = (
        required_features(
            rules
        )
    )

    test_df = load_test_data(
        config,
        model_features,
        rule_features,
    )

    y_true = test_df[
        "binary_target"
    ]

    labels = test_df[
        "Label"
    ]

    print(
        "\nLoading frozen "
        "Random Forest..."
    )

    model = joblib.load(
        config[
            "random_forest"
        ]["model_path"]
    )

    print(
        "Running Random Forest..."
    )

    rf_predictions = model.predict(
        test_df[model_features]
    )

    (
        rf_metrics,
        rf_family,
    ) = evaluate_predictions(
        labels,
        y_true,
        rf_predictions,
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "RANDOM FOREST - "
        "HELD-OUT TEST"
    )
    print("=" * 70)

    print_metrics(
        rf_metrics
    )

    print_attack_family_results(
        rf_family
    )

    print(
        "\nRunning traditional IDS..."
    )

    (
        traditional_predictions,
        _,
        rule_matches,
    ) = apply_rules(
        test_df,
        rules,
    )

    (
        traditional_metrics,
        traditional_family,
    ) = evaluate_predictions(
        labels,
        y_true,
        traditional_predictions,
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "TRADITIONAL IDS - "
        "HELD-OUT TEST"
    )
    print("=" * 70)

    print_metrics(
        traditional_metrics
    )

    print_attack_family_results(
        traditional_family
    )

    print(
        "\n" + "=" * 70
    )
    print("TRADITIONAL RULE MATCHES")
    print("=" * 70)

    for rule, count in (
        rule_matches.items()
    ):
        print(
            f"{rule}: {count:,}"
        )

    agreement = agreement_summary(
        y_true,
        rf_predictions,
        traditional_predictions,
    )

    print(
        "\n" + "=" * 70
    )
    print("DETECTOR AGREEMENT")
    print("=" * 70)

    for key, value in (
        agreement.items()
    ):
        print(
            f"{key}: {value:,}"
        )

    save_results(
        config,
        verified,
        rf_metrics,
        rf_family,
        traditional_metrics,
        traditional_family,
        rule_matches,
        agreement,
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "HELD-OUT TEST COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()