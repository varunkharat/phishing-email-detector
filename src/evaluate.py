"""Evaluate trained models with precision, recall, F1, and confusion matrix."""

from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test) -> dict:
    """Return precision/recall/F1 and confusion matrix for a fitted model."""
    raise NotImplementedError
