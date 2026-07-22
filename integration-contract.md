# Integration Contract — Read This First

This defines exactly what each person's code takes in and produces, so all 4 of you
can work at the same time without waiting on each other. Follow your section's
"input/output" format exactly, and everything will plug together at the end.

Everyone: start with the **dummy data** in your section today. Swap in the real
thing from the person upstream of you once it's ready — that swap should take
minutes, not hours, if you stuck to the format below.

---

## Person 1 — Data (Issue 1)

**Your job:** produce a single CSV that everyone else builds against.

**Output format — save as `data/processed_emails.csv`:**

| column | type | description |
|---|---|---|
| `text` | string | the raw/cleaned email body |
| `label` | int | `1` = phishing, `0` = legitimate |

That's it — just two columns. Keep it that simple so nobody downstream has to
guess column names.

**What "done" looks like:** a CSV with a few hundred+ rows, roughly labeled,
committed to `data/` (check file size — if it's huge, mention it in your PR so we
decide whether to `.gitignore` it and share via a Drive link instead).

**Nobody is blocked waiting on you.** Everyone else uses fake data until you're done.

---

## Person 2 — Feature Engineering (Issue 2)

**Start today with this dummy dataset** — don't wait for Person 1:

```python
import pandas as pd

dummy_data = pd.DataFrame({
    "text": [
        "Please verify your account now, click here immediately",
        "Hey, are we still on for lunch tomorrow?",
        "URGENT: your account will be suspended, act now",
        "Attached is the quarterly report you asked for",
        "Confirm your password at http://192.168.1.1/login",
        "Thanks for the update, talk soon",
    ],
    "label": [1, 0, 1, 0, 1, 0],
})
```

**Your job:** write functions matching these exact signatures in
`src/features.py` and `src/preprocessing.py`:

```python
# src/preprocessing.py
def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Returns a fitted-ready TfidfVectorizer (caller will call .fit_transform())."""

# src/features.py
def has_suspicious_link(email_text: str) -> int:
    """Returns 1 if a suspicious link is found, else 0."""

def urgency_word_count(email_text: str) -> int:
    """Returns count of urgency phrases found."""

def sender_mismatch(display_name: str, sender_address: str) -> int:
    """Returns 1 if mismatched, else 0. OK to stub with a simple heuristic."""
```

**Output format — a single combined feature matrix:**

Write one function that ties it together:

```python
def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Input: df with 'text' and 'label' columns (matches Person 1's format).
    Output: (X, y) where
      X = combined feature matrix (TF-IDF + custom features), shape (n_samples, n_features)
      y = df['label'].values
    """
```

**What "done" looks like:** running `build_feature_matrix(dummy_data)` returns
`X, y` with no errors. Once Person 1's real CSV lands, swap `dummy_data` for
`pd.read_csv("data/processed_emails.csv")` — nothing else should need to change.

---

## Person 3 — Model Training & Evaluation (Issue 3)

**Start today with fake feature data** — don't wait for Person 2:

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=20, random_state=42)
```

**Your job:** write these in `src/train.py` and `src/evaluate.py`:

```python
# src/train.py
def train_model(model_name: str, X_train, y_train):
    """model_name in {'logistic_regression', 'naive_bayes', 'random_forest'}.
    Returns the fitted model."""

# src/evaluate.py
def evaluate_model(model, X_test, y_test) -> dict:
    """Returns {'precision': ..., 'recall': ..., 'f1': ..., 'confusion_matrix': ...}"""
```

**Output format:** save the best model with:

```python
import joblib
joblib.dump(best_model, "src/trained_model.joblib")
```

**What "done" looks like:** all three models train and evaluate cleanly on the
`make_classification` fake data, with a documented comparison (README table or
notebook). Once Person 2's real `build_feature_matrix()` is ready, swap the fake
`X, y` for the real ones — training/eval code shouldn't need to change.

---

## Person 4 — Explanation + Interface (Issue 4)

**Start today with a fake model** — don't wait for Person 3:

```python
class FakeModel:
    def predict(self, X):
        return [1]  # always says "phishing" for now
    def predict_proba(self, X):
        return [[0.13, 0.87]]

model = FakeModel()
```

**Your job:** build the full `app.py` UI and `src/explain.py` against this fake
model so the whole interface — text box, button, verdict display, explanation
section — works end-to-end today.

```python
# src/explain.py
def explain_prediction(model, vectorizer, email_text: str) -> list[str]:
    """Returns a list of strings — the top phrases/words that drove the verdict.
    OK to return a hardcoded example list while using FakeModel."""
```

**What "done" looks like:** running `streamlit run app.py` locally shows a
working page — paste text, click button, see a verdict and explanation (even if
fake). Once Person 3's `src/trained_model.joblib` exists, load the real model
instead of `FakeModel` — the rest of your UI code shouldn't need to change.

```python
import joblib
model = joblib.load("src/trained_model.joblib")
```

---

## End-of-week integration checklist

When real pieces are ready, each swap is a one-line change:

- [ ] Person 2 swaps dummy DataFrame → `pd.read_csv("data/processed_emails.csv")`
- [ ] Person 3 swaps `make_classification()` → `build_feature_matrix(real_df)`
- [ ] Person 4 swaps `FakeModel()` → `joblib.load("src/trained_model.joblib")`

If everyone kept to the input/output shapes above, these three swaps are the
*entire* integration step — no rewriting required.

## If a format needs to change

Formats will probably need small tweaks once real data shows up — that's normal.
Just post in your team chat before changing a shape/column name that others
depend on, so nobody's code silently breaks.
