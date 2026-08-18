from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROCESSED_DATASET = Path(
    "data/processed/cicids2017_clean.csv"
)

SPLIT_DATA_DIR = Path(
    "data/processed/splits"
)

RANDOM_STATE = 42


def load_processed_dataset() -> pd.DataFrame:
    """
    Load the validated, processed CIC-IDS2017 dataset.
    """

    if not PROCESSED_DATASET.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: "
            f"{PROCESSED_DATASET.resolve()}"
        )

    print("Loading processed dataset...")

    df = pd.read_csv(
        PROCESSED_DATASET
    )

    print(
        f"Loaded {len(df):,} records."
    )

    return df


def split_dataset(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split CIC-IDS2017 into deterministic,
    stratified train, validation and test
    partitions.

    Final proportions:

    Training:   70%
    Validation: 15%
    Test:       15%

    Stratification is performed using the
    binary IDS target.
    """

    if "binary_target" not in df.columns:
        raise ValueError(
            "binary_target column not found."
        )

    # ---------------------------------------------------------
    # Stage 1
    # 70% training / 30% temporary
    # ---------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["binary_target"],
    )

    # ---------------------------------------------------------
    # Stage 2
    # Split temporary data equally:
    # 15% validation / 15% test
    # ---------------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["binary_target"],
    )

    # Reset indexes so each output partition
    # has a clean deterministic index.
    train_df = train_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return (
        train_df,
        validation_df,
        test_df,
    )


def report_partition(
    name: str,
    df: pd.DataFrame,
) -> None:
    """
    Print partition size and binary class distribution.
    """

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Records: {len(df):,}"
    )

    class_counts = (
        df["binary_target"]
        .value_counts()
        .sort_index()
    )

    class_percentages = (
        df["binary_target"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    for target, count in class_counts.items():

        class_name = (
            "BENIGN"
            if target == 0
            else "ATTACK"
        )

        percentage = (
            class_percentages[target]
        )

        print(
            f"{class_name}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )


def save_partitions(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Save the three experimental partitions.

    These files are generated artefacts and remain
    outside Git version control.
    """

    SPLIT_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "train.csv": train_df,
        "validation.csv": validation_df,
        "test.csv": test_df,
    }

    for filename, partition in outputs.items():

        output_path = (
            SPLIT_DATA_DIR
            / filename
        )

        print(
            f"Saving {filename}..."
        )

        partition.to_csv(
            output_path,
            index=False,
        )


def main() -> None:

    print("=" * 70)
    print("CIC-IDS2017 DATASET SPLIT")
    print("=" * 70)

    df = load_processed_dataset()

    (
        train_df,
        validation_df,
        test_df,
    ) = split_dataset(df)

    report_partition(
        "TRAINING PARTITION",
        train_df,
    )

    report_partition(
        "VALIDATION PARTITION",
        validation_df,
    )

    report_partition(
        "TEST PARTITION",
        test_df,
    )

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    total = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print(
        f"Original records: "
        f"{len(df):,}"
    )

    print(
        f"Partition records: "
        f"{total:,}"
    )

    print(
        f"Training: "
        f"{len(train_df):,} "
        f"({len(train_df) / total * 100:.2f}%)"
    )

    print(
        f"Validation: "
        f"{len(validation_df):,} "
        f"({len(validation_df) / total * 100:.2f}%)"
    )

    print(
        f"Test: "
        f"{len(test_df):,} "
        f"({len(test_df) / total * 100:.2f}%)"
    )

    if total != len(df):
        raise RuntimeError(
            "Partition sizes do not equal "
            "original dataset size."
        )

    save_partitions(
        train_df,
        validation_df,
        test_df,
    )

    print(
        "\nDataset split complete."
    )


if __name__ == "__main__":
    main()