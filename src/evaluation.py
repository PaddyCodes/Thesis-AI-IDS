from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true, predictions):
    # Keep the metrics in one place so both IDS approaches
    # are measured in exactly the same way
    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(metrics):
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.6f}"
    )
    print(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )
    print(
        f"Recall:    "
        f"{metrics['recall']:.6f}"
    )
    print(
        f"F1-score:  "
        f"{metrics['f1']:.6f}"
    )
    print(
        f"FPR:       "
        f"{metrics['fpr']:.6f}"
    )
    print(
        f"FNR:       "
        f"{metrics['fnr']:.6f}"
    )

    print("\nConfusion matrix:")
    print(f"TN: {metrics['tn']:,}")
    print(f"FP: {metrics['fp']:,}")
    print(f"FN: {metrics['fn']:,}")
    print(f"TP: {metrics['tp']:,}")


def attack_family_recall(
    labels,
    y_true,
    predictions,
):
    # Overall metrics can hide attacks that the model struggles
    # with, so check every attack family separately
    results = []

    attack_labels = sorted(
        label
        for label in labels.unique()
        if label != "BENIGN"
    )

    for attack in attack_labels:
        mask = labels == attack

        total = int(mask.sum())

        detected = int(
            (predictions[mask] == 1).sum()
        )

        missed = total - detected

        recall = (
            detected / total
            if total > 0
            else 0
        )

        results.append(
            {
                "attack": attack,
                "samples": total,
                "detected": detected,
                "missed": missed,
                "recall": recall,
            }
        )

    return results


def print_attack_family_results(results):
    print("\n" + "=" * 70)
    print("RECALL BY ATTACK TYPE")
    print("=" * 70)

    for result in results:
        print(
            f"{result['attack']}: "
            f"{result['recall']:.4f} "
            f"({result['detected']:,}/"
            f"{result['samples']:,}, "
            f"missed {result['missed']:,})"
        )