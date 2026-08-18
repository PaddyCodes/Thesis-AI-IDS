from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import (
    CONSTANT_FEATURES,
    DUPLICATE_FEATURES,
)


PROCESSED_DATASET = Path(
    "data/processed/cicids2017_clean.csv"
)


def validate_dataset() -> None:
    """
    Perform final validation of the processed CIC-IDS2017 dataset.

    This script does not modify the dataset.
    """

    print("=" * 70)
    print("CIC-IDS2017 FINAL DATA VALIDATION")
    print("=" * 70)

    print("\nLoading processed dataset...")

    if not PROCESSED_DATASET.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: "
            f"{PROCESSED_DATASET.resolve()}"
        )

    df = pd.read_csv(PROCESSED_DATASET)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    problems = []

    # ---------------------------------------------------------
    # 1. Column-name validation
    # ---------------------------------------------------------

    print("\n[1] COLUMN NAME CHECK")

    whitespace_columns = [
        column
        for column in df.columns
        if column != column.strip()
    ]

    duplicate_column_names = (
        df.columns[df.columns.duplicated()].tolist()
    )

    print(
        "Columns containing leading/trailing whitespace: "
        f"{len(whitespace_columns)}"
    )

    print(
        f"Duplicate column names: "
        f"{len(duplicate_column_names)}"
    )

    if whitespace_columns:
        print("Whitespace columns:")
        for column in whitespace_columns:
            print(f"  {repr(column)}")

        problems.append(
            "Whitespace remains in column names."
        )

    if duplicate_column_names:
        print("Duplicate names:")
        for column in duplicate_column_names:
            print(f"  {column}")

        problems.append(
            "Duplicate column names detected."
        )

    # ---------------------------------------------------------
    # 2. Missing values
    # ---------------------------------------------------------

    print("\n[2] MISSING VALUE CHECK")

    missing_by_column = df.isna().sum()

    missing_by_column = missing_by_column[
        missing_by_column > 0
    ]

    total_missing = int(
        missing_by_column.sum()
    )

    print(
        f"Total missing values: "
        f"{total_missing:,}"
    )

    if total_missing > 0:
        print(missing_by_column)

        problems.append(
            "Missing values remain."
        )

    # ---------------------------------------------------------
    # 3. Infinite values
    # ---------------------------------------------------------

    print("\n[3] INFINITY CHECK")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    infinity_by_column = (
        np.isinf(numeric_df)
        .sum()
    )

    infinity_by_column = infinity_by_column[
        infinity_by_column > 0
    ]

    total_infinity = int(
        infinity_by_column.sum()
    )

    print(
        f"Total infinite values: "
        f"{total_infinity:,}"
    )

    if total_infinity > 0:
        print(infinity_by_column)

        problems.append(
            "Infinite values remain."
        )

    # ---------------------------------------------------------
    # 4. Specific flow-rate validation
    # ---------------------------------------------------------

    print("\n[4] FLOW RATE FEATURE CHECK")

    rate_features = [
        "Flow Bytes/s",
        "Flow Packets/s",
    ]

    for feature in rate_features:

        if feature not in df.columns:
            print(
                f"WARNING: {feature} not found"
            )

            problems.append(
                f"Expected feature missing: {feature}"
            )

            continue

        missing = int(
            df[feature].isna().sum()
        )

        infinite = int(
            np.isinf(
                df[feature].to_numpy()
            ).sum()
        )

        print(
            f"{feature}: "
            f"NaN={missing:,}, "
            f"Infinite={infinite:,}"
        )

        if missing > 0 or infinite > 0:
            problems.append(
                f"Invalid values remain in "
                f"{feature}."
            )

    # ---------------------------------------------------------
    # 5. Duplicate row validation
    # ---------------------------------------------------------

    print("\n[5] DUPLICATE ROW CHECK")

    duplicates = int(
        df.duplicated().sum()
    )

    print(
        f"Exact duplicate rows: "
        f"{duplicates:,}"
    )

    if duplicates > 0:
        problems.append(
            "Duplicate records remain."
        )

    # ---------------------------------------------------------
    # 6. Binary target validation
    # ---------------------------------------------------------

    print("\n[6] BINARY TARGET CHECK")

    if "binary_target" not in df.columns:
        problems.append(
            "binary_target column is missing."
        )

        print(
            "ERROR: binary_target column missing"
        )

    else:
        target_values = sorted(
            df["binary_target"]
            .unique()
            .tolist()
        )

        print(
            f"Target values: "
            f"{target_values}"
        )

        if target_values != [0, 1]:
            problems.append(
                "binary_target contains "
                "unexpected values."
            )

        if "Label" not in df.columns:
            problems.append(
                "Label column is missing."
            )

        else:
            expected_target = (
                df["Label"] != "BENIGN"
            ).astype(int)

            target_mismatches = int(
                (
                    df["binary_target"]
                    != expected_target
                ).sum()
            )

            print(
                "Label/target inconsistencies: "
                f"{target_mismatches:,}"
            )

            if target_mismatches > 0:
                problems.append(
                    "Label and binary_target "
                    "disagree."
                )

    # ---------------------------------------------------------
    # 7. Label integrity validation
    # ---------------------------------------------------------

    print("\n[7] LABEL INTEGRITY CHECK")

    if "Label" not in df.columns:
        print(
            "ERROR: Label column missing"
        )

        problems.append(
            "Label column is missing."
        )

    else:
        corrupted_labels = df[
            df["Label"].str.contains(
                "\ufffd",
                regex=False,
                na=False,
            )
        ]

        whitespace_labels = df[
            df["Label"]
            != df["Label"].str.strip()
        ]

        print(
            "Labels containing Unicode "
            "replacement character: "
            f"{len(corrupted_labels):,}"
        )

        print(
            "Labels containing surrounding "
            "whitespace: "
            f"{len(whitespace_labels):,}"
        )

        if len(corrupted_labels) > 0:
            problems.append(
                "Corrupted label characters remain."
            )

        if len(whitespace_labels) > 0:
            problems.append(
                "Whitespace remains in labels."
            )

    # ---------------------------------------------------------
    # 8. Feature datatype validation
    # ---------------------------------------------------------

    print("\n[8] FEATURE DATATYPE CHECK")

    feature_columns = [
        column
        for column in df.columns
        if column not in {
            "Label",
            "binary_target",
        }
    ]

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    print(
        "Non-numeric model features: "
        f"{len(non_numeric_features)}"
    )

    if non_numeric_features:
        print("Non-numeric:")
        for column in non_numeric_features:
            print(f"  {column}")

        problems.append(
            "Non-numeric model features "
            "detected."
        )

    # ---------------------------------------------------------
    # 9. Constant feature check
    # ---------------------------------------------------------

    print("\n[9] CONSTANT FEATURE CHECK")

    unique_counts = (
        df[feature_columns]
        .nunique(dropna=False)
    )

    constant_features = (
        unique_counts[
            unique_counts <= 1
        ]
        .index
        .tolist()
    )

    print(
        f"Constant features: "
        f"{len(constant_features)}"
    )

    for feature in constant_features:
        print(f"  {feature}")

    if constant_features:
        problems.append(
            "Constant features remain."
        )

    # ---------------------------------------------------------
    # 10. Known redundant feature check
    # ---------------------------------------------------------

    print("\n[10] REDUNDANT FEATURE CHECK")

    known_redundant_features = (
        CONSTANT_FEATURES
        + DUPLICATE_FEATURES
    )

    remaining_redundant_features = [
        feature
        for feature in known_redundant_features
        if feature in df.columns
    ]

    print(
        "Known redundant features remaining: "
        f"{len(remaining_redundant_features)}"
    )

    if remaining_redundant_features:
        for feature in remaining_redundant_features:
            print(f"  {feature}")

        problems.append(
            "Known redundant features remain."
        )

    # ---------------------------------------------------------
    # 11. Expected feature count
    # ---------------------------------------------------------

    print("\n[11] FEATURE COUNT CHECK")

    expected_feature_count = 69

    print(
        f"Expected model features: "
        f"{expected_feature_count}"
    )

    print(
        f"Actual model features: "
        f"{len(feature_columns)}"
    )

    if len(feature_columns) != expected_feature_count:
        problems.append(
            "Unexpected number of model features."
        )

    # ---------------------------------------------------------
    # 12. Class distribution
    # ---------------------------------------------------------

    print("\n[12] CLASS DISTRIBUTION")

    if "binary_target" in df.columns:

        class_counts = (
            df["binary_target"]
            .value_counts()
            .sort_index()
        )

        class_percentages = (
            df["binary_target"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        )

        for target, count in class_counts.items():

            label = (
                "BENIGN"
                if target == 0
                else "ATTACK"
            )

            percentage = (
                class_percentages[target]
            )

            print(
                f"{label}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    if problems:

        print("VALIDATION FAILED")

        print("\nProblems detected:")

        for problem in problems:
            print(f"  - {problem}")

    else:

        print("VALIDATION PASSED")

        print(
            "No blocking data-quality "
            "problems detected."
        )


if __name__ == "__main__":
    validate_dataset()