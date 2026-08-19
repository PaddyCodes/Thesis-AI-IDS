from pathlib import Path
import hashlib
import json

import pandas as pd


CONFIG_FILE = Path(
    "configs/ai_attack_experiment.json"
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


def load_config(path=CONFIG_FILE):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"AI experiment config not found: "
            f"{path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    required = {
        "experiment",
        "source_test",
        "generation",
        "immutable_features",
        "timing_features",
        "inverse_rate_features",
    }

    missing = (
        required
        - set(config)
    )

    if missing:
        raise ValueError(
            "AI experiment config is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    return config


def verify_source_test(config):
    source = config[
        "source_test"
    ]

    path = Path(
        source["path"]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Source test set not found: "
            f"{path.resolve()}"
        )

    actual_hash = sha256_file(
        path
    )

    expected_hash = (
        source["sha256"].upper()
    )

    if actual_hash != expected_hash:
        raise ValueError(
            "Source test hash mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    return actual_hash


def validate_source_frame(
    df,
    config,
):
    source = config[
        "source_test"
    ]

    if (
        len(df)
        != source["expected_rows"]
    ):
        raise ValueError(
            "Unexpected source row count."
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
        != source["expected_benign"]
    ):
        raise ValueError(
            "Unexpected benign count."
        )

    if (
        attacks
        != source["expected_attacks"]
    ):
        raise ValueError(
            "Unexpected attack count."
        )


def load_source_data(config):
    source_path = Path(
        config[
            "source_test"
        ]["path"]
    )

    print(
        "Loading source test timing data..."
    )

    df = pd.read_csv(
        source_path,
        usecols=[
            "Label",
            "binary_target",
            "Flow Duration",
        ],
    )

    validate_source_frame(
        df,
        config,
    )

    return df


def summarise_group(
    df,
    scope,
    group,
):
    values = df[
        "Flow Duration"
    ]

    quantiles = values.quantile(
        [
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ]
    )

    return {
        "scope": scope,
        "group": group,
        "count": int(
            len(values)
        ),
        "minimum": float(
            values.min()
        ),
        "q05": float(
            quantiles.loc[0.05]
        ),
        "q25": float(
            quantiles.loc[0.25]
        ),
        "median": float(
            quantiles.loc[0.50]
        ),
        "q75": float(
            quantiles.loc[0.75]
        ),
        "q95": float(
            quantiles.loc[0.95]
        ),
        "maximum": float(
            values.max()
        ),
    }


def build_timing_summary(df):
    rows = []

    benign = df[
        df["binary_target"] == 0
    ]

    rows.append(
        summarise_group(
            benign,
            "benign",
            "BENIGN",
        )
    )

    attack_labels = sorted(
        label
        for label in df[
            "Label"
        ].unique()
        if label != "BENIGN"
    )

    for label in attack_labels:
        family = df[
            df["Label"] == label
        ]

        rows.append(
            summarise_group(
                family,
                "attack_family",
                label,
            )
        )

    return pd.DataFrame(
        rows
    )


def build_ai_request(
    config,
    summary,
    source_hash,
):
    generation = config[
        "generation"
    ]

    benign_row = (
        summary[
            summary["scope"]
            == "benign"
        ]
        .iloc[0]
        .to_dict()
    )

    attack_rows = (
        summary[
            summary["scope"]
            == "attack_family"
        ]
        .drop(
            columns=["scope"]
        )
        .to_dict(
            orient="records"
        )
    )

    return {
        "request_version": 1,
        "purpose": (
            "Defensive evaluation of "
            "AI-assisted timing modification "
            "of existing malicious flow "
            "feature vectors."
        ),
        "strategy": (
            "detector_independent_"
            "timing_morphing"
        ),
        "source_test_sha256":
            source_hash,
        "detector_information_provided":
            False,
        "allowed_scale": {
            "minimum":
                generation[
                    "minimum_scale"
                ],
            "maximum":
                generation[
                    "maximum_scale"
                ],
        },
        "transformation": {
            "description": (
                "A positive timing scale "
                "will later be applied to "
                "all timing features for a "
                "record. Rate features will "
                "be scaled inversely."
            ),
            "immutable_features":
                config[
                    "immutable_features"
                ],
            "timing_features":
                config[
                    "timing_features"
                ],
            "inverse_rate_features":
                config[
                    "inverse_rate_features"
                ],
        },
        "constraints": [
            (
                "Use only the supplied "
                "timing distributions."
            ),
            (
                "Do not assume access to "
                "either intrusion detector."
            ),
            (
                "Do not use detector rules, "
                "feature importance, "
                "predictions or performance."
            ),
            (
                "Do not alter attack labels "
                "or attack-family identity."
            ),
            (
                "Do not propose exploitation "
                "steps or new attacks."
            ),
            (
                "Choose one scale interval "
                "for every attack family."
            ),
            (
                "Every scale must remain "
                "inside the allowed range."
            ),
            (
                "Do not return an interval "
                "of exactly 1.0 to 1.0."
            ),
        ],
        "benign_reference":
            benign_row,
        "attack_families":
            attack_rows,
        "required_output": {
            "plan_version": 1,
            "strategy": (
                "detector_independent_"
                "timing_morphing"
            ),
            "families": [
                {
                    "label":
                        "<exact attack label>",
                    "min_scale":
                        "<number>",
                    "max_scale":
                        "<number>",
                    "rationale":
                        "<concise timing-only "
                        "reason>",
                }
            ],
        },
    }


def save_outputs(
    config,
    summary,
    request,
):
    summary_path = Path(
        config[
            "generation"
        ]["summary_path"]
    )

    request_path = Path(
        config[
            "generation"
        ]["request_path"]
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    with request_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            request,
            file,
            indent=4,
        )

    print("\nSaved:")
    print(summary_path)
    print(request_path)


def main():
    print("=" * 70)
    print(
        "PREPARE AI-ASSISTED "
        "ATTACK CONDITION"
    )
    print("=" * 70)

    config = load_config()

    print(
        "Verifying original held-out "
        "test set..."
    )

    source_hash = (
        verify_source_test(
            config
        )
    )

    print(
        "Source test hash verified."
    )

    df = load_source_data(
        config
    )

    print(
        f"Source records: {len(df):,}"
    )

    summary = (
        build_timing_summary(
            df
        )
    )

    request = build_ai_request(
        config,
        summary,
        source_hash,
    )

    save_outputs(
        config,
        summary,
        request,
    )

    print(
        "\nNo detector rules, model "
        "outputs or detector performance "
        "have been included."
    )

    print(
        "No attack records have been "
        "modified yet."
    )


if __name__ == "__main__":
    main()