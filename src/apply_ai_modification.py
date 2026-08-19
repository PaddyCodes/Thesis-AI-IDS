from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


EXPERIMENT_CONFIG = Path(
    "configs/ai_attack_experiment.json"
)

PLAN_FILE = Path(
    "configs/ai_modification_plan.json"
)

EXPECTED_PLAN_HASH = (
    "A7A06D886B1D80F44DC18EAD8BF29E48"
    "B12AAAAD8D5442DFF64F9BC9D4D3FC20"
)

OUTPUT_FILE = Path(
    "data/processed/ai_modified_test.csv"
)

SCALE_LOG_FILE = Path(
    "outputs/ai_generation/"
    "ai_transformation_scales.csv"
)

SUMMARY_FILE = Path(
    "outputs/ai_generation/"
    "ai_transformation_summary.csv"
)

METADATA_FILE = Path(
    "outputs/ai_generation/"
    "ai_transformation_metadata.json"
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


def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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

    actual = sha256_file(path)

    if actual != expected_hash.upper():
        raise ValueError(
            f"{name} hash mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual}"
        )

    return actual


def validate_plan(
    plan,
    config,
    source_labels=None,
):
    if plan.get("plan_version") != 1:
        raise ValueError(
            "Unexpected AI plan version."
        )

    expected_strategy = (
        "detector_independent_"
        "timing_morphing"
    )

    if (
        plan.get("strategy")
        != expected_strategy
    ):
        raise ValueError(
            "Unexpected AI plan strategy."
        )

    families = plan.get(
        "families",
        []
    )

    if not families:
        raise ValueError(
            "AI plan contains no families."
        )

    allowed_min = float(
        config["generation"][
            "minimum_scale"
        ]
    )

    allowed_max = float(
        config["generation"][
            "maximum_scale"
        ]
    )

    labels = []

    for family in families:
        label = family.get("label")

        if not label:
            raise ValueError(
                "AI plan contains an "
                "empty attack label."
            )

        labels.append(label)

        minimum = float(
            family["min_scale"]
        )

        maximum = float(
            family["max_scale"]
        )

        if not (
            allowed_min
            <= minimum
            <= allowed_max
        ):
            raise ValueError(
                f"{label} min_scale is "
                "outside the allowed range."
            )

        if not (
            allowed_min
            <= maximum
            <= allowed_max
        ):
            raise ValueError(
                f"{label} max_scale is "
                "outside the allowed range."
            )

        if minimum > maximum:
            raise ValueError(
                f"{label} has a reversed "
                "scale interval."
            )

        if (
            minimum == 1.0
            and maximum == 1.0
        ):
            raise ValueError(
                f"{label} has a no-op "
                "scale interval."
            )

        if not str(
            family.get(
                "rationale",
                "",
            )
        ).strip():
            raise ValueError(
                f"{label} has no rationale."
            )

    if len(labels) != len(
        set(labels)
    ):
        raise ValueError(
            "AI plan contains duplicate "
            "attack labels."
        )

    if source_labels is not None:
        if set(labels) != set(
            source_labels
        ):
            missing = (
                set(source_labels)
                - set(labels)
            )

            extra = (
                set(labels)
                - set(source_labels)
            )

            raise ValueError(
                "AI plan attack-family "
                "coverage mismatch. "
                f"Missing: {sorted(missing)}. "
                f"Extra: {sorted(extra)}."
            )


def plan_map(plan):
    return {
        family["label"]: family
        for family in plan["families"]
    }


def required_features(config):
    return list(
        dict.fromkeys(
            [
                "Label",
                "binary_target",
                *config[
                    "timing_features"
                ],
                *config[
                    "inverse_rate_features"
                ],
            ]
        )
    )


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

    if int(
        counts.get(0, 0)
    ) != source["expected_benign"]:
        raise ValueError(
            "Unexpected benign count."
        )

    if int(
        counts.get(1, 0)
    ) != source["expected_attacks"]:
        raise ValueError(
            "Unexpected attack count."
        )

    missing = [
        feature
        for feature
        in required_features(config)
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Source dataset is missing "
            f"required features: {missing}"
        )

    maximum_duration = float(
        config["generation"][
            "maximum_flow_duration"
        ]
    )

    if (
        df["Flow Duration"].max()
        > maximum_duration
    ):
        raise ValueError(
            "Source dataset already "
            "exceeds the configured "
            "maximum flow duration."
        )


def choose_scales(
    df,
    config,
    plan,
):
    seed = int(
        config["generation"][
            "random_seed"
        ]
    )

    maximum_duration = float(
        config["generation"][
            "maximum_flow_duration"
        ]
    )

    rng = np.random.default_rng(
        seed
    )

    attack_mask = (
        df["binary_target"] == 1
    )

    source_labels = set(
        df.loc[
            attack_mask,
            "Label",
        ].unique()
    )

    validate_plan(
        plan,
        config,
        source_labels,
    )

    sampled_scales = pd.Series(
        1.0,
        index=df.index,
        dtype="float64",
    )

    applied_scales = pd.Series(
        1.0,
        index=df.index,
        dtype="float64",
    )

    constrained = pd.Series(
        False,
        index=df.index,
        dtype="bool",
    )

    audit_rows = []

    for family in plan["families"]:
        label = family["label"]

        mask = (
            attack_mask
            & (df["Label"] == label)
        )

        indexes = df.index[
            mask
        ]

        count = len(indexes)

        minimum = float(
            family["min_scale"]
        )

        maximum = float(
            family["max_scale"]
        )

        sampled = rng.uniform(
            minimum,
            maximum,
            size=count,
        )

        durations = (
            df.loc[
                indexes,
                "Flow Duration",
            ]
            .astype("float64")
            .to_numpy()
        )

        feasible_max = np.full(
            count,
            np.inf,
            dtype="float64",
        )

        positive_duration = (
            durations > 0
        )

        feasible_max[
            positive_duration
        ] = (
            maximum_duration
            / durations[
                positive_duration
            ]
        )

        # The plan remains fixed. Only the
        # effective per-row scale is reduced
        # if an expansion would break the
        # configured duration ceiling.
        applied = np.minimum(
            sampled,
            feasible_max,
        )

        was_constrained = (
            applied < sampled
        )

        sampled_scales.loc[
            indexes
        ] = sampled

        applied_scales.loc[
            indexes
        ] = applied

        constrained.loc[
            indexes
        ] = was_constrained

        for (
            row_index,
            sampled_scale,
            applied_scale,
            limited,
        ) in zip(
            indexes,
            sampled,
            applied,
            was_constrained,
        ):
            audit_rows.append(
                {
                    "source_row":
                        int(row_index),
                    "label":
                        label,
                    "sampled_scale":
                        float(
                            sampled_scale
                        ),
                    "applied_scale":
                        float(
                            applied_scale
                        ),
                    "constrained":
                        bool(limited),
                }
            )

    audit = pd.DataFrame(
        audit_rows
    )

    return (
        sampled_scales,
        applied_scales,
        constrained,
        audit,
    )


def transform_frame(
    df,
    config,
    plan,
):
    original = df.copy(
        deep=True
    )

    transformed = df.copy(
        deep=True
    )

    (
        sampled_scales,
        applied_scales,
        constrained,
        audit,
    ) = choose_scales(
        original,
        config,
        plan,
    )

    attack_mask = (
        original[
            "binary_target"
        ] == 1
    )

    timing_features = config[
        "timing_features"
    ]

    rate_features = config[
        "inverse_rate_features"
    ]

    for feature in timing_features:
        transformed.loc[
            attack_mask,
            feature,
        ] = (
            original.loc[
                attack_mask,
                feature,
            ].astype("float64")
            * applied_scales.loc[
                attack_mask
            ]
        )

    for feature in rate_features:
        transformed.loc[
            attack_mask,
            feature,
        ] = (
            original.loc[
                attack_mask,
                feature,
            ].astype("float64")
            / applied_scales.loc[
                attack_mask
            ]
        )

    validate_transformation(
        original,
        transformed,
        config,
        applied_scales,
    )

    return (
        transformed,
        sampled_scales,
        applied_scales,
        constrained,
        audit,
    )


def validate_transformation(
    original,
    transformed,
    config,
    applied_scales,
):
    if list(
        original.columns
    ) != list(
        transformed.columns
    ):
        raise ValueError(
            "Dataset columns changed "
            "during transformation."
        )

    if len(original) != len(
        transformed
    ):
        raise ValueError(
            "Dataset row count changed "
            "during transformation."
        )

    benign_mask = (
        original[
            "binary_target"
        ] == 0
    )

    if not original.loc[
        benign_mask
    ].equals(
        transformed.loc[
            benign_mask
        ]
    ):
        raise ValueError(
            "Benign records changed."
        )

    modified_features = set(
        config["timing_features"]
        + config[
            "inverse_rate_features"
        ]
    )

    unchanged_features = [
        column
        for column
        in original.columns
        if column
        not in modified_features
    ]

    if not original[
        unchanged_features
    ].equals(
        transformed[
            unchanged_features
        ]
    ):
        raise ValueError(
            "A non-modifiable feature "
            "was changed."
        )

    if not original[
        "Label"
    ].equals(
        transformed[
            "Label"
        ]
    ):
        raise ValueError(
            "Attack labels changed."
        )

    if not original[
        "binary_target"
    ].equals(
        transformed[
            "binary_target"
        ]
    ):
        raise ValueError(
            "Binary targets changed."
        )

    maximum_duration = float(
        config["generation"][
            "maximum_flow_duration"
        ]
    )

    if (
        transformed[
            "Flow Duration"
        ].max()
        > maximum_duration
        + 1e-9
    ):
        raise ValueError(
            "Transformed flow duration "
            "exceeds the configured limit."
        )

    attack_mask = (
        transformed[
            "binary_target"
        ] == 1
    )

    if (
        applied_scales.loc[
            attack_mask
        ] <= 0
    ).any():
        raise ValueError(
            "Invalid transformation scale."
        )

    numeric_features = (
        config["timing_features"]
        + config[
            "inverse_rate_features"
        ]
    )

    values = transformed.loc[
        attack_mask,
        numeric_features,
    ].to_numpy(
        dtype="float64"
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Transformation introduced "
            "NaN or infinite values."
        )


def build_summary(
    audit,
    plan,
):
    plan_lookup = plan_map(
        plan
    )

    rows = []

    for label, group in (
        audit.groupby(
            "label",
            sort=False,
        )
    ):
        rule = plan_lookup[
            label
        ]

        rows.append(
            {
                "label": label,
                "records":
                    int(len(group)),
                "requested_min_scale":
                    float(
                        rule["min_scale"]
                    ),
                "requested_max_scale":
                    float(
                        rule["max_scale"]
                    ),
                "sampled_scale_mean":
                    float(
                        group[
                            "sampled_scale"
                        ].mean()
                    ),
                "applied_scale_mean":
                    float(
                        group[
                            "applied_scale"
                        ].mean()
                    ),
                "applied_scale_min":
                    float(
                        group[
                            "applied_scale"
                        ].min()
                    ),
                "applied_scale_max":
                    float(
                        group[
                            "applied_scale"
                        ].max()
                    ),
                "constrained_records":
                    int(
                        group[
                            "constrained"
                        ].sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def ensure_not_generated():
    existing = [
        path
        for path in [
            OUTPUT_FILE,
            SCALE_LOG_FILE,
            SUMMARY_FILE,
            METADATA_FILE,
        ]
        if path.exists()
    ]

    if existing:
        raise RuntimeError(
            "AI-modified condition already "
            "exists. Refusing to overwrite: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )


def save_results(
    transformed,
    audit,
    summary,
    config,
    source_hash,
    plan_hash,
):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SCALE_LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transformed.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    audit.to_csv(
        SCALE_LOG_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    output_hash = sha256_file(
        OUTPUT_FILE
    )

    scale_log_hash = sha256_file(
        SCALE_LOG_FILE
    )

    metadata = {
        "experiment":
            config["experiment"],
        "generated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "random_seed":
            int(
                config["generation"][
                    "random_seed"
                ]
            ),
        "source_test_sha256":
            source_hash,
        "ai_plan_sha256":
            plan_hash,
        "output_dataset": {
            "path":
                str(OUTPUT_FILE),
            "sha256":
                output_hash,
            "rows":
                int(
                    len(transformed)
                ),
            "benign":
                int(
                    (
                        transformed[
                            "binary_target"
                        ] == 0
                    ).sum()
                ),
            "attacks":
                int(
                    (
                        transformed[
                            "binary_target"
                        ] == 1
                    ).sum()
                ),
        },
        "scale_audit": {
            "path":
                str(SCALE_LOG_FILE),
            "sha256":
                scale_log_hash,
            "records":
                int(len(audit)),
            "constrained_records":
                int(
                    audit[
                        "constrained"
                    ].sum()
                ),
        },
        "timing_features":
            config[
                "timing_features"
            ],
        "inverse_rate_features":
            config[
                "inverse_rate_features"
            ],
        "maximum_flow_duration":
            float(
                config[
                    "generation"
                ][
                    "maximum_flow_duration"
                ]
            ),
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

    return (
        output_hash,
        scale_log_hash,
    )


def main():
    print("=" * 70)
    print(
        "APPLY AI-ASSISTED "
        "TIMING MODIFICATION"
    )
    print("=" * 70)

    ensure_not_generated()

    config = load_json(
        EXPERIMENT_CONFIG
    )

    plan = load_json(
        PLAN_FILE
    )

    print(
        "Verifying frozen AI plan..."
    )

    plan_hash = verify_hash(
        PLAN_FILE,
        EXPECTED_PLAN_HASH,
        "AI modification plan",
    )

    print(
        "AI plan hash verified."
    )

    source_path = Path(
        config[
            "source_test"
        ]["path"]
    )

    print(
        "Verifying source test set..."
    )

    source_hash = verify_hash(
        source_path,
        config[
            "source_test"
        ]["sha256"],
        "Original held-out test set",
    )

    print(
        "Source test hash verified."
    )

    print(
        "\nLoading original held-out "
        "test data..."
    )

    df = pd.read_csv(
        source_path
    )

    validate_source_frame(
        df,
        config,
    )

    print(
        f"Records loaded: {len(df):,}"
    )

    attack_labels = set(
        df.loc[
            df["binary_target"] == 1,
            "Label",
        ].unique()
    )

    validate_plan(
        plan,
        config,
        attack_labels,
    )

    print(
        "AI plan validation passed."
    )

    print(
        "\nApplying deterministic "
        "timing transformation..."
    )

    (
        transformed,
        _,
        _,
        _,
        audit,
    ) = transform_frame(
        df,
        config,
        plan,
    )

    summary = build_summary(
        audit,
        plan,
    )

    print(
        "Transformation integrity "
        "checks passed."
    )

    print(
        "\nTransformation summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    (
        output_hash,
        scale_log_hash,
    ) = save_results(
        transformed,
        audit,
        summary,
        config,
        source_hash,
        plan_hash,
    )

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(SCALE_LOG_FILE)
    print(SUMMARY_FILE)
    print(METADATA_FILE)

    print(
        "\nModified dataset SHA-256:"
    )
    print(output_hash)

    print(
        "\nScale audit SHA-256:"
    )
    print(scale_log_hash)

    print(
        "\nBenign records were "
        "preserved unchanged."
    )

    print(
        "Labels and non-timing "
        "features were preserved."
    )

    print(
        "Neither IDS has been "
        "evaluated on this condition."
    )


if __name__ == "__main__":
    main()