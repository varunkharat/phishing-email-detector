"""
Model evaluation for the phishing detector  (Person 3 / Issue 3).

Public function (matches the integration contract exactly):
    evaluate_model(model, X_test, y_test) -> dict with
        {'precision', 'recall', 'f1', 'confusion_matrix'}

WHY NOT ACCURACY:
Phishing datasets are imbalanced — most emails are legitimate. A model that
labels EVERY email "legitimate" can score 95% accuracy while catching zero
phishing emails. So we report:
    precision = of the emails we flagged, how many really were phishing?
                (low precision = annoying false alarms)
    recall    = of all the real phishing emails, how many did we catch?
                (low recall = dangerous misses — this is the one that matters most)
    F1        = the balance between the two; used to pick the winning model.
"""

import numpy as np


def evaluate_model(model, X_test, y_test) -> dict:
    """Score a fitted model on the held-out test set.

    Returns a dict:
        {'precision': float, 'recall': float, 'f1': float,
         'confusion_matrix': np.ndarray, 'accuracy': float}

    'accuracy' is included for the report's comparison table, but the model
    choice is made on F1 (see train.py).
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = model.predict(X_test)

    # pos_label=1 => phishing is the "positive" class we care about catching.
    # zero_division=0 keeps things from crashing if a model predicts one class only.
    return {
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def print_confusion_matrix(cm):
    """Print the confusion matrix with labels, so it's readable in the report.

    Layout:
                       predicted legit   predicted phishing
        actual legit         TN                  FP
        actual phishing      FN                  TP

    FN (false negatives) are the dangerous ones: real phishing that slipped
    through. FP are legitimate mail wrongly flagged — annoying, not dangerous.
    """
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        print(cm)
        return

    tn, fp, fn, tp = cm.ravel()
    print("                    predicted legit   predicted phishing")
    print(f"  actual legit       {tn:>10}        {fp:>10}")
    print(f"  actual phishing    {fn:>10}        {tp:>10}")
    print(f"  -> {fn} phishing email(s) missed, {fp} false alarm(s)")


def print_comparison_table(results: dict):
    """Print the 3-model comparison table for the README / report.

    `results` maps model_name -> the dict returned by evaluate_model().
    """
    print("\n" + "=" * 68)
    print("MODEL COMPARISON")
    print("=" * 68)
    header = f"{'model':<22}{'precision':>11}{'recall':>10}{'F1':>10}{'accuracy':>12}"
    print(header)
    print("-" * 68)
    for name, r in results.items():
        print(
            f"{name:<22}{r['precision']:>11.3f}{r['recall']:>10.3f}"
            f"{r['f1']:>10.3f}{r['accuracy']:>12.3f}"
        )
    print("-" * 68)

    for name, r in results.items():
        print(f"\nConfusion matrix — {name}:")
        print_confusion_matrix(r["confusion_matrix"])


def markdown_comparison_table(results: dict) -> str:
    """Same table as Markdown, ready to paste into the README or the report."""
    lines = [
        "| Model | Precision | Recall | F1 | Accuracy |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} "
            f"| {r['f1']:.3f} | {r['accuracy']:.3f} |"
        )
    return "\n".join(lines)
