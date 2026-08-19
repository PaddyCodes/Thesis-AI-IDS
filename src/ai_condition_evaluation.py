from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import pandas as pd

from src.evaluation import (
    print_attack_family_results,
    print_metrics,
)
from src.final_evaluation import (
    agreement_summary,
    evaluate_predictions,
    load_feature_names,
    serialise_metrics,
    sha256_file,
    validate_test_frame,
    verify_hash,
)
from src.traditional_ids import (
    apply_rules,
    load_rules,
    required_features,
)


CONFIG_FILE = Path(
    "configs/ai_condition_experiment.json"
)

OUTPUT_DIR = Path(
    "outputs/metrics"
)

METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "fnr",
    "tn",
    "fp",
    "fn",
    "tp",
]

OUTPUT_FILES = {
    "rf_metrics":
        OUTPUT_DIR
        / "ai_condition_random_forest_metrics.csv",
    "traditional_metrics":
        OUTPUT_DIR
        / "ai_condition_traditional_ids_metrics.csv",
    "comparison":
        OUTPUT_DIR
        / "ai_condition_comparison.csv",
    "rf_family":
        OUTPUT_DIR
        / "ai_condition_random_forest_attack_recall.csv",
    "traditional_family":
        OUTPUT_DIR
        / "ai_condition_traditional_ids_attack_recall.csv",
    "rule_matches":
        OUTPUT_DIR
        / "ai_condition_traditional_rule_matches.csv",
    "agreement":
        OUTPUT_DIR
        / "ai_condition_agreement.csv",
    "disagreements":
        OUTPUT_DIR
        / "ai_condition_disagreements.csv",
    "original_comparison":
        OUTPUT_DIR
        / "ai_condition_vs_original_metrics.csv",
    "metadata":
        OUTPUT_DIR
        / "ai_condition_metadata.json",
}


def load_experiment_config(
    path=CONFIG_FILE,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Experiment config not found: "
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
        "ai_provenance",
        "original_results",
    }

    missing = (
        required_sections
        - set(config)
    )

    if missing:
        raise ValueError(
            "Experiment config is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    return config


def ensure_not_already_run(
    output_files=None,
):
    if output_files is None:
        output_files = (
            OUTPUT_FILES.values()
        )

    existing = [
        Path(path)
        for path in output_files
        if Path(path).exists()
    ]

    if existing:
        names = "\n".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "AI-assisted condition results "
            "already exist. Refusing to "
            "overwrite:\n"
            f"{names}"
        )


def verify_frozen_inputs(config):
    dataset = config[
        "dataset"
    ]

    model = config[
        "random_forest"
    ]

    traditional = config[
        "traditional_ids"
    ]

    ai_provenance = config[
        "ai_provenance"
    ]

    print(
        "Verifying frozen experiment "
        "artifacts..."
    )

    dataset_hash = verify_hash(
        dataset["path"],
        dataset["sha256"],
        "AI-assisted held-out dataset",
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

    plan_hash = verify_hash(
        ai_provenance["plan_path"],
        ai_provenance[
            "plan_sha256"
        ],
        "AI modification plan",
    )

    scale_audit_hash = verify_hash(
        ai_provenance[
            "scale_audit_path"
        ],
        ai_provenance[
            "scale_audit_sha256"
        ],
        "AI transformation scale audit",
    )

    features = load_feature_names(
        model["features_path"]
    )

    expected_count = model[
        "expected_feature_count"
    ]

    if len(features) != expected_count:
        raise ValueError(
            "Unexpected model feature "
            "count. "
            f"Expected {expected_count}, "
            f"found {len(features)}."
        )

    print(
        "All frozen artifact hashes "
        "verified."
    )

    return {
        "dataset_sha256":
            dataset_hash,
        "model_sha256":
            model_hash,
        "features_sha256":
            features_hash,
        "rules_sha256":
            rules_hash,
        "plan_sha256":
            plan_hash,
        "scale_audit_sha256":
            scale_audit_hash,
        "model_features":
            features,
    }


def load_test_data(
    config,
    model_features,
    rule_features,
):
    path = Path(
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
        "\nLoading AI-assisted "
        "held-out data..."
    )

    df = pd.read_csv(
        path,
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
        "AI-assisted test condition "
        "integrity checks passed."
    )

    return df


def read_original_metrics(
    path,
    expected_detector,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Original result file not "
            f"found: {path.resolve()}"
        )

    frame = pd.read_csv(
        path
    )

    if len(frame) != 1:
        raise ValueError(
            "Original metric file must "
            "contain exactly one row."
        )

    if "detector" not in frame:
        raise ValueError(
            "Original metric file has no "
            "detector column."
        )

    detector = str(
        frame.iloc[0][
            "detector"
        ]
    )

    if detector != expected_detector:
        raise ValueError(
            "Unexpected detector in "
            "original metric file. "
            f"Expected {expected_detector}, "
            f"found {detector}."
        )

    missing = [
        key
        for key in METRIC_KEYS
        if key not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Original metric file is "
            "missing: "
            + ", ".join(missing)
        )

    return {
        key: float(
            frame.iloc[0][key]
        )
        for key in METRIC_KEYS
    }


def load_original_results(config):
    original = config[
        "original_results"
    ]

    rf_path = Path(
        original[
            "random_forest_metrics_path"
        ]
    )

    traditional_path = Path(
        original[
            "traditional_ids_metrics_path"
        ]
    )

    results = {
        "random_forest":
            read_original_metrics(
                rf_path,
                "random_forest",
            ),
        "traditional_rule_ids":
            read_original_metrics(
                traditional_path,
                "traditional_rule_ids",
            ),
    }

    hashes = {
        "random_forest_metrics_sha256":
            sha256_file(
                rf_path
            ),
        "traditional_ids_metrics_sha256":
            sha256_file(
                traditional_path
            ),
    }

    return (
        results,
        hashes,
    )


def build_metric_comparison(
    original_results,
    rf_metrics,
    traditional_metrics,
):
    ai_results = {
        "random_forest":
            rf_metrics,
        "traditional_rule_ids":
            traditional_metrics,
    }

    rows = []

    for detector in [
        "random_forest",
        "traditional_rule_ids",
    ]:
        for metric in METRIC_KEYS:
            original_value = float(
                original_results[
                    detector
                ][metric]
            )

            ai_value = float(
                ai_results[
                    detector
                ][metric]
            )

            rows.append(
                {
                    "detector":
                        detector,
                    "metric":
                        metric,
                    "original_value":
                        original_value,
                    "ai_assisted_value":
                        ai_value,
                    "change":
                        (
                            ai_value
                            - original_value
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_disagreement_rows(
    labels,
    y_true,
    rf_predictions,
    traditional_predictions,
):
    labels = pd.Series(
        labels
    ).reset_index(drop=True)

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
        len(labels)
        == len(truth)
        == len(rf)
        == len(traditional)
    ):
        raise ValueError(
            "Prediction lengths do not "
            "match."
        )

    frame = pd.DataFrame(
        {
            "source_row":
                range(len(truth)),
            "label":
                labels,
            "binary_target":
                truth,
            "random_forest_prediction":
                rf,
            "traditional_prediction":
                traditional,
        }
    )

    frame[
        "random_forest_correct"
    ] = (
        frame[
            "random_forest_prediction"
        ]
        == frame["binary_target"]
    )

    frame[
        "traditional_correct"
    ] = (
        frame[
            "traditional_prediction"
        ]
        == frame["binary_target"]
    )

    return frame.loc[
        frame[
            "random_forest_prediction"
        ]
        != frame[
            "traditional_prediction"
        ]
    ].copy()


def save_results(
    config,
    verified,
    rf_metrics,
    rf_family,
    traditional_metrics,
    traditional_family,
    rule_matches,
    agreement,
    disagreements,
    original_comparison,
    original_hashes,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rf_row = {
        "detector":
            "random_forest",
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
        OUTPUT_FILES[
            "rf_metrics"
        ],
        index=False,
    )

    pd.DataFrame(
        [traditional_row]
    ).to_csv(
        OUTPUT_FILES[
            "traditional_metrics"
        ],
        index=False,
    )

    pd.DataFrame(
        [
            rf_row,
            traditional_row,
        ]
    ).to_csv(
        OUTPUT_FILES[
            "comparison"
        ],
        index=False,
    )

    pd.DataFrame(
        rf_family
    ).to_csv(
        OUTPUT_FILES[
            "rf_family"
        ],
        index=False,
    )

    pd.DataFrame(
        traditional_family
    ).to_csv(
        OUTPUT_FILES[
            "traditional_family"
        ],
        index=False,
    )

    pd.DataFrame(
        [
            {
                "rule":
                    rule,
                "matches":
                    count,
            }
            for rule, count
            in rule_matches.items()
        ]
    ).to_csv(
        OUTPUT_FILES[
            "rule_matches"
        ],
        index=False,
    )

    pd.DataFrame(
        [agreement]
    ).to_csv(
        OUTPUT_FILES[
            "agreement"
        ],
        index=False,
    )

    disagreements.to_csv(
        OUTPUT_FILES[
            "disagreements"
        ],
        index=False,
    )

    original_comparison.to_csv(
        OUTPUT_FILES[
            "original_comparison"
        ],
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
                config[
                    "dataset"
                ]["path"],
            "sha256":
                verified[
                    "dataset_sha256"
                ],
            "records":
                config[
                    "dataset"
                ][
                    "expected_rows"
                ],
            "benign":
                config[
                    "dataset"
                ][
                    "expected_benign"
                ],
            "attacks":
                config[
                    "dataset"
                ][
                    "expected_attacks"
                ],
        },
        "ai_provenance": {
            "plan_sha256":
                verified[
                    "plan_sha256"
                ],
            "scale_audit_sha256":
                verified[
                    "scale_audit_sha256"
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
            "rule_matches": {
                key: int(value)
                for key, value
                in rule_matches.items()
            },
        },
        "agreement":
            agreement,
        "disagreement_records":
            int(
                len(
                    disagreements
                )
            ),
        "original_result_hashes":
            original_hashes,
    }

    with OUTPUT_FILES[
        "metadata"
    ].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        "\nAI-assisted condition "
        "results saved."
    )


def main():
    print("=" * 70)
    print(
        "AI-ASSISTED HELD-OUT "
        "TEST EVALUATION"
    )
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
        test_df[
            model_features
        ]
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
        "AI-ASSISTED CONDITION"
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
        "AI-ASSISTED CONDITION"
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
    print(
        "TRADITIONAL RULE MATCHES"
    )
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

    disagreements = (
        build_disagreement_rows(
            labels,
            y_true,
            rf_predictions,
            traditional_predictions,
        )
    )

    # Original results are loaded only after
    # both frozen detectors have produced
    # their AI-condition predictions.
    (
        original_results,
        original_hashes,
    ) = load_original_results(
        config
    )

    original_comparison = (
        build_metric_comparison(
            original_results,
            rf_metrics,
            traditional_metrics,
        )
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
        disagreements,
        original_comparison,
        original_hashes,
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "AI-ASSISTED HELD-OUT "
        "TEST COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()