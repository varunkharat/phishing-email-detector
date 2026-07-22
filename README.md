# Phishing Email Detector

A supervised machine learning system that classifies emails as **phishing** or **legitimate** based on their text content.

## Overview

We combine standard NLP features (TF-IDF) with hand-crafted security signals
(suspicious links, urgency language, sender-address mismatches) and compare
several classifiers (Logistic Regression, Naive Bayes, Random Forest) to find
the best-performing model. Evaluation prioritizes precision/recall/F1 over
raw accuracy, since phishing datasets are imbalanced.

## Project structure

```
phishing-detector/
├── data/                  # raw + processed datasets (not committed if large — see .gitignore)
├── notebooks/             # exploratory analysis
├── src/
│   ├── preprocessing.py   # text cleaning + TF-IDF vectorization
│   ├── features.py        # custom security-specific feature extraction
│   ├── train.py           # trains and compares classifiers
│   ├── evaluate.py        # precision / recall / F1, confusion matrix
│   └── explain.py         # highlights phrases/links that drove a prediction
├── app.py                 # Streamlit interface (paste email → get verdict)
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

## Team workflow

- Work on feature branches, not directly on `main`: `git checkout -b feature/tfidf-pipeline`
- Open a Pull Request into `main` when a piece is ready for review
- Use the GitHub **Issues** tab to track tasks and avoid duplicate work
- Use the GitHub **Projects** board (Kanban view) to see who's working on what

## Suggested task split (4 people)

1. **Data** — sourcing and cleaning the phishing/legitimate email dataset
2. **Feature engineering** — TF-IDF pipeline + custom security signals (`features.py`)
3. **Modeling & evaluation** — train/compare classifiers, precision/recall/F1 (`train.py`, `evaluate.py`)
4. **Explanation & interface** — flagged-phrase highlighting + Streamlit app (`explain.py`, `app.py`)

## Status

🚧 Early development — model training pipeline not yet implemented.
