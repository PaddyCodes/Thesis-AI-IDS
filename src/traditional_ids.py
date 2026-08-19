from pathlib import Path
import hashlib
import json

import pandas as pd

from src.evaluation import (
    attack_family_recall,
    calculate_metrics,
    print_attack_family_results,
    print_metrics,
)


RULE_FILE = Path(
    "configs/traditional_rules.json"
)

VALIDATION_FILE = Path(
    "data/processed/splits/validation.csv"
)

METRICS_DIR = Path(
    "outputs/metrics"
)


SUPPORTED_OPERATORS = {
    "eq",
    "le",
    "lt",
    "ge",
    "gt",
    "between",
}


def load_rules(rule_file=RULE_FILE):
    if not rule_file.exists():
        raise FileNotFoundError(
            f"Rule file not found: "
            f"{rule_file.resolve()}"
        )

    with rule_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    if "rules" not in config:
        raise ValueError(
            "Rule configuration does not "
            "contain a rules section."
        )

    if not config["rules"]:
        raise ValueError(
            "No traditional IDS rules found."
        )

    validate_rules(
        config["rules"]
    )

    return config


def validate_rules(rules):
    rule_names = set()

    for rule in rules:
        name = rule.get("name")

        if not name:
            raise ValueError(
                "Every rule needs a name."
            )

        if name in rule_names:
            raise ValueError(
                f"Duplicate rule name: {name}"
            )

        rule_names.add(name)

        conditions = rule.get(
            "conditions",
            []
        )

        if not conditions:
            raise ValueError(
                f"{name} has no conditions."
            )

        for condition in conditions:
            feature = condition.get(
                "feature"
            )

            operator = condition.get(
                "operator"
            )

            if not feature:
                raise ValueError(
                    f"{name} has a condition "
                    "without a feature."
                )

            if operator not in (
                SUPPORTED_OPERATORS
            ):
                raise ValueError(
                    f"{name} uses unsupported "
                    f"operator: {operator}"
                )

            if operator == "between":
                if (
                    "min" not in condition
                    or "max" not in condition
                ):
                    raise ValueError(
                        f"{name} has an invalid "
                        "between condition."
                    )

                if (
                    condition["min"]
                    > condition["max"]
                ):
                    raise ValueError(
                        f"{name} has a reversed "
                        "between range."
                    )

            elif "value" not in condition:
                raise ValueError(
                    f"{name} condition on "
                    f"{feature} has no value."
                )


def required_features(rules):
    return sorted(
        {
            condition["feature"]
            for rule in rules
            for condition
            in rule["conditions"]
        }
    )


def condition_mask(
    df,
    condition,
):
    feature = condition["feature"]
    operator = condition["operator"]

    if feature not in df.columns:
        raise ValueError(
            f"Required feature not found: "
            f"{feature}"
        )

    values = df[feature]

    if operator == "eq":
        return (
            values
            == condition["value"]
        )

    if operator == "le":
        return (
            values
            <= condition["value"]
        )

    if operator == "lt":
        return (
            values
            < condition["value"]
        )

    if operator == "ge":
        return (
            values
            >= condition["value"]
        )

    if operator == "gt":
        return (
            values
            > condition["value"]
        )

    if operator == "between":
        return values.between(
            condition["min"],
            condition["max"],
            inclusive="both",
        )

    raise ValueError(
        f"Unsupported operator: "
        f"{operator}"
    )


def rule_mask(
    df,
    rule,
):
    # Every condition in a rule has to match
    # before the rule can raise an alert.
    mask = pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )

    for condition in rule[
        "conditions"
    ]:
        mask &= condition_mask(
            df,
            condition,
        )

    return mask


def apply_rules(
    df,
    rules,
):
    # Rules are ORed together for the final
    # benign/attack prediction.
    predictions = pd.Series(
        0,
        index=df.index,
        dtype="int8",
    )

    first_match = pd.Series(
        "",
        index=df.index,
        dtype="object",
    )

    match_counts = {}

    for rule in rules:
        mask = rule_mask(
            df,
            rule,
        )

        match_counts[
            rule["name"]
        ] = int(
            mask.sum()
        )

        # Keep the first matching rule so an alert
        # still has a simple explanation.
        new_matches = (
            mask
            & (predictions == 0)
        )

        first_match.loc[
            new_matches
        ] = rule["name"]

        predictions.loc[
            mask
        ] = 1

    return (
        predictions.to_numpy(),
        first_match,
        match_counts,
    )


def load_validation_data(
    features,
):
    # Validation is only loaded after the
    # rule set has been fixed.
    columns = [
        "Label",
        "binary_target",
        *features,
    ]

    print(
        "Loading validation data..."
    )

    df = pd.read_csv(
        VALIDATION_FILE,
        usecols=columns,
    )

    print(
        f"Validation records: "
        f"{len(df):,}"
    )

    return df


def rule_file_hash(
    rule_file=RULE_FILE,
):
    hasher = hashlib.sha256()

    with rule_file.open(
        "rb"
    ) as file:
        for chunk in iter(
            lambda: file.read(65536),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def save_results(
    config,
    metrics,
    family_results,
    match_counts,
):
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            {
                "detector":
                    "traditional_rule_ids",
                **metrics,
            }
        ]
    ).to_csv(
        METRICS_DIR
        / "traditional_ids_validation_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        family_results
    ).to_csv(
        METRICS_DIR
        / "traditional_ids_attack_recall.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "rule": rule,
                "matches": count,
            }
            for rule, count
            in match_counts.items()
        ]
    ).to_csv(
        METRICS_DIR
        / "traditional_ids_rule_matches.csv",
        index=False,
    )

    metadata = {
        "detector": (
            "deterministic_flow_rule_ids"
        ),
        "rule_version": config[
            "version"
        ],
        "rule_count": len(
            config["rules"]
        ),
        "rule_file": str(
            RULE_FILE
        ),
        "rule_file_sha256":
            rule_file_hash(),
        "validation_records":
            int(
                metrics["tn"]
                + metrics["fp"]
                + metrics["fn"]
                + metrics["tp"]
            ),
        "validation_metrics": {
            key: (
                int(value)
                if key in {
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                }
                else float(value)
            )
            for key, value
            in metrics.items()
        },
    }

    metadata_file = (
        METRICS_DIR
        / "traditional_ids_metadata.json"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        "\nSaved traditional IDS "
        "validation outputs."
    )


def main():
    print("=" * 70)
    print("TRADITIONAL RULE-BASED IDS")
    print("=" * 70)

    config = load_rules()

    rules = config["rules"]

    features = required_features(
        rules
    )

    print(
        f"Rules loaded: "
        f"{len(rules)}"
    )

    print(
        f"Features used by rules: "
        f"{len(features)}"
    )

    validation_df = (
        load_validation_data(
            features
        )
    )

    print(
        "\nApplying deterministic rules..."
    )

    (
        predictions,
        first_match,
        match_counts,
    ) = apply_rules(
        validation_df,
        rules,
    )

    metrics = calculate_metrics(
        validation_df[
            "binary_target"
        ],
        predictions,
    )

    family_results = (
        attack_family_recall(
            validation_df["Label"],
            validation_df[
                "binary_target"
            ],
            predictions,
        )
    )

    print_metrics(
        metrics
    )

    print_attack_family_results(
        family_results
    )

    print("\n" + "=" * 70)
    print("RULE MATCH COUNTS")
    print("=" * 70)

    for rule, count in (
        match_counts.items()
    ):
        print(
            f"{rule}: {count:,}"
        )

    save_results(
        config,
        metrics,
        family_results,
        match_counts,
    )

    print("\n" + "=" * 70)
    print("TRADITIONAL IDS COMPLETE")
    print("=" * 70)

    print(
        "The held-out test partition "
        "has not been accessed."
    )


if __name__ == "__main__":
    main()