from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import find_csv_files


RAW_DATA_DIR = Path("data/raw/cicids2017")
PROCESSED_DATA_DIR = Path("data/processed")

CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

DUPLICATE_FEATURES = [
    "Fwd Header Length.1",
]

def remove_redundant_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove structural features that contain no useful independent
    information.

    Constant features contain the same value for every record.
    Fwd Header Length.1 is an exact duplicate of Fwd Header Length.
    """
    df = df.copy()

    features_to_remove = (
        CONSTANT_FEATURES
        + DUPLICATE_FEATURES
    )

    missing_features = [
        feature
        for feature in features_to_remove
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Expected redundant features were not found: "
            f"{missing_features}"
        )

    df.drop(
        columns=features_to_remove,
        inplace=True,
    )

    report = {
        "constant_features_removed":
            len(CONSTANT_FEATURES),

        "duplicate_features_removed":
            len(DUPLICATE_FEATURES),

        "total_features_removed":
            len(features_to_remove),
    }

    return df, report

def load_raw_dataset() -> pd.DataFrame:
    """
    Load and combine all raw CIC-IDS2017 CSV files.

    The source CSV files are not modified.
    """
    csv_files = find_csv_files(RAW_DATA_DIR)

    frames = []

    for file_path in csv_files:
        print(f"Loading {file_path.name}")
        df = pd.read_csv(file_path)
        frames.append(df)

    combined_df = pd.concat(
        frames,
        ignore_index=True
    )

    print(f"\nCombined rows: {len(combined_df):,}")
    print(f"Combined columns: {len(combined_df.columns)}")

    return combined_df


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from CIC-IDS2017 column names.
    """
    df = df.copy()

    df.columns = [
        column.strip()
        for column in df.columns
    ]

    return df


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise whitespace and known encoding artefacts in the
    CIC-IDS2017 attack labels.

    The original attack categories are retained for later
    per-attack-family analysis.
    """
    df = df.copy()

    if "Label" not in df.columns:
        raise ValueError("Expected 'Label' column was not found.")

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    label_corrections = {
        "Web Attack � Brute Force": "Web Attack - Brute Force",
        "Web Attack � XSS": "Web Attack - XSS",
        "Web Attack � Sql Injection": "Web Attack - SQL Injection",
    }

    df["Label"] = df["Label"].replace(label_corrections)

    return df


def remove_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Replace positive/negative infinity with NaN and remove records
    containing missing or invalid values.
    """
    df = df.copy()

    original_rows = len(df)

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    rows_with_invalid_values = int(
        df.isna().any(axis=1).sum()
    )

    df.dropna(inplace=True)

    report = {
        "rows_before": original_rows,
        "invalid_rows_removed": rows_with_invalid_values,
        "rows_after": len(df),
    }

    return df, report


def remove_duplicate_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove exact duplicate records.
    """
    df = df.copy()

    rows_before = len(df)
    duplicates = int(df.duplicated().sum())

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    report = {
        "rows_before": rows_before,
        "duplicates_removed": duplicates,
        "rows_after": len(df),
    }

    return df, report


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the primary binary IDS target.

    BENIGN = 0
    ATTACK = 1

    The original multiclass Label field is preserved.
    """
    df = df.copy()

    df["binary_target"] = (
        df["Label"] != "BENIGN"
    ).astype(int)

    return df


def preprocess_dataset() -> pd.DataFrame:
    print("=" * 70)
    print("CIC-IDS2017 PREPROCESSING")
    print("=" * 70)

    df = load_raw_dataset()

    print("\nNormalising column names...")
    df = normalise_column_names(df)

    print("Normalising labels...")
    df = normalise_labels(df)

    print("Removing invalid values...")
    df, invalid_report = remove_invalid_rows(df)

    print(
        f"Invalid rows removed: "
        f"{invalid_report['invalid_rows_removed']:,}"
    )

    print("Removing duplicate rows...")
    df, duplicate_report = remove_duplicate_rows(df)

    print(
        f"Duplicate rows removed: "
        f"{duplicate_report['duplicates_removed']:,}"
    )

    print("Removing redundant structural features...")
    df, feature_report = remove_redundant_features(df)

    print(
        f"Constant features removed: "
        f"{feature_report['constant_features_removed']}"
    )

    print(
        f"Duplicate features removed: "
        f"{feature_report['duplicate_features_removed']}"
    )

    print("Creating binary target...")
    df = create_binary_target(df)

    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print(f"Final rows: {len(df):,}")
    print(f"Final columns: {len(df.columns)}")

    print("\nBinary target distribution:")
    print(df["binary_target"].value_counts())

    print("\nOriginal attack labels:")
    print(df["Label"].value_counts())

    return df


def main():
    df = preprocess_dataset()

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DATA_DIR /
        "cicids2017_clean.csv"
    )

    print(f"\nSaving processed dataset to: {output_path}")

    df.to_csv(
        output_path,
        index=False
    )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()