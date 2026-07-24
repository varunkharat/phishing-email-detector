# Phishing Email Detector

A supervised machine learning system that classifies emails as **phishing** or **legitimate** based on their text content.

## Overview

We combine standard NLP features (TF-IDF) with hand-crafted security signals
(suspicious links, urgency language, sender-address mismatches) and compare
several classifiers (Logistic Regression, Naive Bayes, Random Forest) to find
the best-performing model. Evaluation prioritizes precision/recall/F1 over
raw accuracy, since phishing datasets are imbalanced.

## Results

Trained and evaluated on **82,485 real emails** (52% phishing / 48% legitimate),
80/20 train/test split:

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.981 | 0.985 | 0.983 | 0.982 |
| Naive Bayes | 0.984 | 0.925 | 0.953 | 0.953 |
| **Random Forest** | **0.986** | **0.987** | **0.987** | **0.986** |

**Random Forest wins on F1** and is the model the app uses. On the held-out
test set (16,497 emails), it missed **108 phishing emails** and raised
**119 false alarms** — both under 1.5% of the test set.

We picked the winner by F1 rather than accuracy, since a model that always
predicts "legitimate" would score deceptively well on accuracy alone in an
imbalanced dataset like this.

## Project structure

```
phishing-detector/
├── data/                  # raw + processed datasets (not committed — see .gitignore)
├── models/
│   └── tfidf_vectorizer.joblib   # fitted TF-IDF vectorizer
├── src/
│   ├── preprocessing.py   # text cleaning + TF-IDF vectorization
│   ├── features.py        # custom security-specific feature extraction
│   ├── train.py           # trains and compares classifiers
│   ├── evaluate.py        # precision / recall / F1, confusion matrix
│   ├── explain.py         # highlights phrases/links that drove a prediction
│   └── trained_model.joblib   # NOT committed (too large) — regenerate locally, see below
├── app.py                 # Streamlit interface (paste email → get verdict)
├── prepare_data.py        # converts raw dataset into data/processed_emails.csv
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd phishing-detector
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running it

The repo already includes a small baseline model + vectorizer, so the app
works out of the box with no extra setup:

```bash
streamlit run app.py
```

**To reproduce the real Random Forest results yourself** (the trained model
file is too large for GitHub, so it isn't committed):

1. Get `raw_emails.csv` from a teammate (too large for GitHub — shared separately) and place it at `data/raw_emails.csv`
2. Run:
   ```bash
   python3 prepare_data.py
   python -m src.train
   ```
   This regenerates `data/processed_emails.csv` and `src/trained_model.joblib`.
   Training is deterministic (fixed random seed), so you'll get the same
   results shown above.

## Known limitation

`explain.py` can show which specific words drove a prediction for linear
models (Logistic Regression, Naive Bayes) using their coefficients. For
Random Forest, we tried using SHAP for the same purpose, but `TreeExplainer`
did not perform well on a model this size (200 trees, 5,000+ features) —
in practice it hung indefinitely with no error. SHAP is disabled for now;
explanations fall back to our rule-based checks (suspicious links, urgency
phrases, sender mismatch), which work reliably. Worth revisiting with a
smaller/approximate SHAP configuration if time allows.

## Team workflow

- Work on feature branches, not directly on `main`: `git checkout -b feature/tfidf-pipeline`
- Open a Pull Request into `main` when a piece is ready for review
- Use the GitHub **Issues** tab to track tasks and avoid duplicate work

## Task split (4 people)

1. **Data** — sourced and cleaned the phishing/legitimate email dataset (`prepare_data.py`)
2. **Feature engineering** — TF-IDF pipeline + custom security signals (`features.py`, `preprocessing.py`)
3. **Modeling & evaluation** — trained/compared classifiers, precision/recall/F1 (`train.py`, `evaluate.py`)
4. **Explanation & interface** — flagged-phrase highlighting + Streamlit app (`explain.py`, `app.py`)

## Status

✅ All four components complete and integrated end to end: real dataset →
features → trained model → explanations → working Streamlit interface.
