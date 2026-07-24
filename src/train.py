"""
Model training for the phishing detector  (Person 3 / Issue 3).

Public function (matches the integration contract exactly):
    train_model(model_name, X_train, y_train) -> fitted model

Also includes a runnable script at the bottom that:
  - loads data (real if available, fake otherwise)
  - trains all three models
  - evaluates and compares them
  - saves the best one to src/trained_model.joblib

Run it with:
    python -m src.train
"""

import os

import joblib
import numpy as np

MODEL_PATH = "src/trained_model.joblib"

# The three algorithms named in the project proposal.
VALID_MODELS = ("logistic_regression", "naive_bayes", "random_forest")


def train_model(model_name: str, X_train, y_train):
    """Train one classifier and return the fitted model.

    model_name must be one of:
        'logistic_regression', 'naive_bayes', 'random_forest'

    Notes on the choices:
    - LogisticRegression: fast, works well on sparse TF-IDF text, and exposes
      coef_ so Person 4's explanation feature can show which words mattered.
    - MultinomialNB: the classic text-classification baseline. Needs
      non-negative features, which TF-IDF + our custom counts satisfy.
    - RandomForest: captures feature interactions the linear models miss.
      class_weight='balanced' matters here because phishing datasets are
      imbalanced (far more legitimate emails than phishing ones).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    name = model_name.lower().strip()

    if name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "naive_bayes":
        model = MultinomialNB()
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Expected one of {VALID_MODELS}."
        )

    model.fit(X_train, y_train)
    return model


def save_model(model, path: str = MODEL_PATH):
    """Save the winning model so the Streamlit app can load it."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved best model to {path}")


# ---------------------------------------------------------------------------
# Runnable comparison script
# ---------------------------------------------------------------------------
def load_training_data():
    """Return (X, y, source_label).

    Uses the real dataset if Person 1's CSV exists and Person 2's feature code
    is importable; otherwise falls back to make_classification fake data so this
    file can be developed and tested before those land.
    """
    csv_path = "data/processed_emails.csv"

    if os.path.exists(csv_path):
        try:
            import pandas as pd
            from src.features import build_feature_matrix

            df = pd.read_csv(csv_path)
            X, y = build_feature_matrix(df)
            return X, y, f"real data ({csv_path}, {len(df)} rows)"
        except Exception as exc:
            print(f"[train] could not use real data ({exc}) — falling back to fake.")

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200,
        n_features=20,
        random_state=42,
    )
    # MultinomialNB requires non-negative features; make_classification produces
    # negatives, so shift them up. (Real TF-IDF features are already >= 0.)
    X = X - X.min()
    return X, y, "FAKE data (make_classification) — placeholder only"


def main():
    from sklearn.model_selection import train_test_split

    from src.evaluate import evaluate_model, print_comparison_table

    X, y, source = load_training_data()
    print(f"Training on: {source}")
    print(f"Feature matrix shape: {X.shape}\n")

    # stratify keeps the phishing/legitimate ratio the same in both splits,
    # which matters a lot on imbalanced data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    models = {}
    for name in VALID_MODELS:
        model = train_model(name, X_train, y_train)
        models[name] = model
        results[name] = evaluate_model(model, X_test, y_test)

    print_comparison_table(results)

    # Pick the winner by F1, not accuracy — see evaluate.py for why.
    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest model by F1: {best_name} (F1 = {results[best_name]['f1']:.3f})")
    save_model(models[best_name])

    return results


if __name__ == "__main__":
    main()
