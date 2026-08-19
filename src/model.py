from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)

from src.evaluation import (
    attack_family_recall,
    calculate_metrics,
    print_attack_family_results,
    print_metrics,
)


SPLIT_DIR = Path("data/processed/splits")

TRAIN_FILE = SPLIT_DIR / "train.csv"
VALIDATION_FILE = SPLIT_DIR / "validation.csv"

MODEL_DIR = Path("outputs/models")
METRICS_DIR = Path("outputs/metrics")

RANDOM_STATE = 42
CV_FOLDS = 5
SEARCH_ITERATIONS = 10
SEARCH_JOBS = 2


PARAMETER_DISTRIBUTIONS = {
    "n_estimators": [
        100,
        150,
        200,
    ],
    "max_depth": [
        None,
        20,
        30,
        40,
    ],
    "min_samples_split": [
        2,
        5,
        10,
    ],
    "min_samples_leaf": [
        1,
        2,
        4,
    ],
    "max_features": [
        "sqrt",
        "log2",
        0.5,
    ],
}


def load_model_data():
    # The test set is deliberately left alone until
    # the final experiment
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_FILE)

    print("Loading validation data...")
    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Training records: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation records: "
        f"{len(validation_df):,}"
    )

    return train_df, validation_df


def prepare_features(
    train_df,
    validation_df,
):
    excluded_columns = {
        "Label",
        "binary_target",
    }

    feature_columns = [
        column
        for column in train_df.columns
        if column not in excluded_columns
    ]

    if len(feature_columns) != 69:
        raise ValueError(
            "Expected 69 model features, "
            f"found {len(feature_columns)}."
        )

    validation_features = [
        column
        for column in validation_df.columns
        if column not in excluded_columns
    ]

    if feature_columns != validation_features:
        raise ValueError(
            "Training and validation features "
            "do not match."
        )

    X_train = train_df[feature_columns]
    y_train = train_df["binary_target"]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        "binary_target"
    ]

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    )


def build_baseline_model():
    # Keep the same untuned model we have already
    # measured so the comparison stays fair
    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def build_search():
    # Keep parallelism at the CV level during tuning.
    # Running every forest across every core as well
    # would make resource use unnecessarily heavy.
    search_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=1,
    )

    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return RandomizedSearchCV(
        estimator=search_model,
        param_distributions=(
            PARAMETER_DISTRIBUTIONS
        ),
        n_iter=SEARCH_ITERATIONS,
        scoring="f1",
        n_jobs=SEARCH_JOBS,
        cv=cross_validation,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=2,
        pre_dispatch=SEARCH_JOBS,
        return_train_score=False,
        error_score="raise",
    )


def evaluate_model(
    model,
    X_validation,
    y_validation,
    validation_labels,
):
    predictions = model.predict(
        X_validation
    )

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    family_results = attack_family_recall(
        validation_labels,
        y_validation,
        predictions,
    )

    print_metrics(metrics)

    print_attack_family_results(
        family_results
    )

    return (
        metrics,
        family_results,
    )


def print_search_results(search):
    print("\n" + "=" * 70)
    print("RANDOM FOREST TUNING RESULTS")
    print("=" * 70)

    print(
        f"Candidates tested: "
        f"{SEARCH_ITERATIONS}"
    )

    print(
        f"Cross-validation folds: "
        f"{CV_FOLDS}"
    )

    print(
        f"Best mean CV F1: "
        f"{search.best_score_:.6f}"
    )

    print("\nBest parameters:")

    for parameter, value in (
        search.best_params_.items()
    ):
        print(
            f"  {parameter}: {value}"
        )


def print_model_comparison(
    baseline_metrics,
    tuned_metrics,
):
    print("\n" + "=" * 70)
    print("BASELINE VS TUNED MODEL")
    print("=" * 70)

    metrics_to_compare = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
    ]

    for metric in metrics_to_compare:
        baseline_value = (
            baseline_metrics[metric]
        )

        tuned_value = (
            tuned_metrics[metric]
        )

        difference = (
            tuned_value
            - baseline_value
        )

        print(
            f"{metric.upper():<10} "
            f"Baseline: "
            f"{baseline_value:.6f}   "
            f"Tuned: "
            f"{tuned_value:.6f}   "
            f"Change: "
            f"{difference:+.6f}"
        )


def save_search_results(search):
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.DataFrame(
        search.cv_results_
    )

    columns_to_keep = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_max_features",
    ]

    results = (
        results[columns_to_keep]
        .sort_values(
            "rank_test_score"
        )
    )

    output_file = (
        METRICS_DIR
        / "random_forest_search_results.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved tuning results to "
        f"{output_file}"
    )


def save_validation_results(
    baseline_metrics,
    tuned_metrics,
    baseline_family_results,
    tuned_family_results,
):
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_results = pd.DataFrame(
        [
            {
                "model": "baseline",
                **baseline_metrics,
            },
            {
                "model": "tuned",
                **tuned_metrics,
            },
        ]
    )

    overall_results.to_csv(
        METRICS_DIR
        / "random_forest_validation_metrics.csv",
        index=False,
    )

    baseline_family_df = pd.DataFrame(
        baseline_family_results
    )

    baseline_family_df.insert(
        0,
        "model",
        "baseline",
    )

    tuned_family_df = pd.DataFrame(
        tuned_family_results
    )

    tuned_family_df.insert(
        0,
        "model",
        "tuned",
    )

    family_results = pd.concat(
        [
            baseline_family_df,
            tuned_family_df,
        ],
        ignore_index=True,
    )

    family_results.to_csv(
        METRICS_DIR
        / "random_forest_attack_recall.csv",
        index=False,
    )

    print(
        "Saved validation metrics and "
        "attack-family results."
    )


def save_feature_importance(
    model,
    feature_columns,
):
    importances = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": (
                model.feature_importances_
            ),
        }
    )

    importances = (
        importances
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importances.to_csv(
        METRICS_DIR
        / "random_forest_feature_importance.csv",
        index=False,
    )

    print(
        "Saved feature importance results."
    )


def save_model(
    model,
    feature_columns,
):
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_file = (
        MODEL_DIR
        / "random_forest.joblib"
    )

    joblib.dump(
        model,
        model_file,
    )

    feature_file = (
        MODEL_DIR
        / "random_forest_features.txt"
    )

    feature_file.write_text(
        "\n".join(feature_columns),
        encoding="utf-8",
    )

    print(
        f"Saved final model to "
        f"{model_file}"
    )


def save_metadata(
    search,
    baseline_metrics,
    tuned_metrics,
    train_records,
    validation_records,
    feature_count,
):
    metadata = {
        "classifier": (
            "RandomForestClassifier"
        ),
        "random_state": RANDOM_STATE,
        "scoring": "f1",
        "cv_folds": CV_FOLDS,
        "search_iterations": (
            SEARCH_ITERATIONS
        ),
        "search_jobs": SEARCH_JOBS,
        "class_weight": (
            "balanced_subsample"
        ),
        "train_records": (
            train_records
        ),
        "validation_records": (
            validation_records
        ),
        "feature_count": (
            feature_count
        ),
        "sklearn_version": (
            sklearn_version
        ),
        "best_cv_f1": float(
            search.best_score_
        ),
        "best_parameters": (
            search.best_params_
        ),
        "baseline_validation": {
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
            in baseline_metrics.items()
        },
        "tuned_validation": {
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
            in tuned_metrics.items()
        },
    }

    metadata_file = (
        METRICS_DIR
        / "random_forest_metadata.json"
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
        f"Saved model metadata to "
        f"{metadata_file}"
    )


def main():
    print("=" * 70)
    print("AI IDS - RANDOM FOREST")
    print("=" * 70)

    train_df, validation_df = (
        load_model_data()
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        feature_columns,
    ) = prepare_features(
        train_df,
        validation_df,
    )

    # ---------------------------------------------------------
    # Untuned benchmark
    # ---------------------------------------------------------

    print("\nTraining baseline model...")

    baseline_model = (
        build_baseline_model()
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    print("Baseline training complete.")

    (
        baseline_metrics,
        baseline_family_results,
    ) = evaluate_model(
        baseline_model,
        X_validation,
        y_validation,
        validation_df["Label"],
    )

    # We no longer need the baseline model itself
    # once its validation results have been recorded
    del baseline_model

    # ---------------------------------------------------------
    # Hyperparameter search
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING HYPERPARAMETER SEARCH")
    print("=" * 70)

    search = build_search()

    search.fit(
        X_train,
        y_train,
    )

    print_search_results(
        search
    )

    # RandomizedSearchCV has already refitted the best
    # configuration against the full training partition
    tuned_model = search.best_estimator_

    # The search limits each forest to one worker.
    # Once selected, normal model use can use all cores.
    tuned_model.set_params(
        n_jobs=-1
    )

    # ---------------------------------------------------------
    # Validation of the selected model
    # ---------------------------------------------------------

    print("\nEvaluating tuned model...")

    (
        tuned_metrics,
        tuned_family_results,
    ) = evaluate_model(
        tuned_model,
        X_validation,
        y_validation,
        validation_df["Label"],
    )

    print_model_comparison(
        baseline_metrics,
        tuned_metrics,
    )

    # ---------------------------------------------------------
    # Save everything needed for later evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING MODEL OUTPUTS")
    print("=" * 70)

    save_search_results(
        search
    )

    save_validation_results(
        baseline_metrics,
        tuned_metrics,
        baseline_family_results,
        tuned_family_results,
    )

    save_feature_importance(
        tuned_model,
        feature_columns,
    )

    save_model(
        tuned_model,
        feature_columns,
    )

    save_metadata(
        search,
        baseline_metrics,
        tuned_metrics,
        train_records=len(train_df),
        validation_records=(
            len(validation_df)
        ),
        feature_count=(
            len(feature_columns)
        ),
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST COMPLETE")
    print("=" * 70)

    print(
        "The selected model is ready to "
        "be frozen before final testing."
    )


if __name__ == "__main__":
    main()