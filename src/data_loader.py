from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw/cicids2017")


def find_csv_files(data_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """
    Return all CSV files contained within the raw CIC-IDS2017 directory.
    """
    csv_files = sorted(data_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in: {data_dir.resolve()}"
        )

    return csv_files


def inspect_csv_file(file_path: Path) -> dict:
    """
    Load one CIC-IDS2017 CSV file and return raw-data metadata.
    No preprocessing or modification is performed.
    """
    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(file_path)

    label_column = " Label"

    if label_column not in df.columns:
        raise ValueError(
            f"Expected label column {repr(label_column)} "
            f"not found in {file_path.name}"
        )

    # Class distribution
    label_counts = df[label_column].value_counts(dropna=False)

    # Missing values
    missing_values = int(df.isna().sum().sum())

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Infinite values - check numeric columns only
    numeric_df = df.select_dtypes(include="number")
    infinity_values = int(
        ((numeric_df == float("inf")) | (numeric_df == float("-inf")))
        .sum()
        .sum()
    )

    metadata = {
        "filename": file_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "label_counts": label_counts,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "infinity_values": infinity_values,
    }

    return metadata


def main():
    csv_files = find_csv_files()

    print(f"Found {len(csv_files)} CSV files.")

    summaries = []

    for file_path in csv_files:
        metadata = inspect_csv_file(file_path)
        summaries.append(metadata)

        print(f"Rows: {metadata['rows']:,}")
        print(f"Columns: {metadata['columns']}")
        print(f"Missing values: {metadata['missing_values']:,}")
        print(f"Duplicate rows: {metadata['duplicate_rows']:,}")
        print(f"Infinite values: {metadata['infinity_values']:,}")

        print("Labels:")
        for label, count in metadata["label_counts"].items():
            print(f"  {repr(label)}: {count:,}")

    print("\n" + "=" * 70)
    print("SCHEMA CHECK")
    print("=" * 70)

    reference_columns = summaries[0]["column_names"]

    schemas_match = True

    for summary in summaries:
        if summary["column_names"] != reference_columns:
            schemas_match = False
            print(f"Schema mismatch: {summary['filename']}")

    if schemas_match:
        print("All CSV files have identical column schemas.")
    else:
        print("WARNING: Not all CSV files have identical schemas.")

    print("\nColumns:")
    for index, column in enumerate(reference_columns, start=1):
        print(f"{index:>2}. {repr(column)}")


if __name__ == "__main__":
    main()