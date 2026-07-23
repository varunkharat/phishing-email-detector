"""
Streamlit interface for the phishing detector  (Person 4 / Issue 4).

Run locally with:
    streamlit run app.py

Works TODAY with a placeholder model, and automatically upgrades to the real
model the moment Person 3 saves src/trained_model.joblib (and the vectorizer
from Person 2 exists at models/tfidf_vectorizer.joblib). No code change needed.
"""

import os
import sys

import streamlit as st

# Make `src` importable whether run from the repo root or elsewhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.explain import explain_prediction
from src.features import featurize_email
from src.preprocessing import build_tfidf_vectorizer

MODEL_PATH = "src/trained_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"


# ---------------------------------------------------------------------------
# Placeholder model — used only until the real one is trained.
# ---------------------------------------------------------------------------
class FakeModel:
    """Stand-in so the UI works before Person 3's model exists."""

    def predict(self, X):
        return [1]  # always "phishing" for demo purposes

    def predict_proba(self, X):
        return [[0.13, 0.87]]


@st.cache_resource
def load_pipeline():
    """Load (model, vectorizer, is_real). Falls back to placeholders."""
    import joblib

    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer, True

    # Fallback: fake model + a vectorizer fitted on a tiny sample so that
    # featurize_email() / explain_prediction() still run.
    sample = [
        "please verify your account now click here immediately",
        "hey are we still on for lunch tomorrow",
        "urgent your account will be suspended act now",
        "attached is the quarterly report you asked for",
        "confirm your password at http://192.168.1.1/login",
        "thanks for the update talk soon",
    ]
    vectorizer = build_tfidf_vectorizer(max_features=100)
    # relax min_df for the tiny sample so fitting doesn't error
    vectorizer.set_params(min_df=1)
    vectorizer.fit(sample)
    return FakeModel(), vectorizer, False


def classify(model, vectorizer, text):
    """Return (label:int, phishing_probability:float)."""
    X = featurize_email(text, vectorizer)
    label = int(model.predict(X)[0])
    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception:
        proba = float(label)
    return label, proba


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️")

st.title("🛡️ Phishing Email Detector")
st.write(
    "Paste an email below and the model will judge whether it's **phishing** or "
    "**legitimate**, and explain why."
)

model, vectorizer, is_real = load_pipeline()

if not is_real:
    st.warning(
        "⚠️ Running on a **placeholder model** — the real trained model isn't "
        "available yet, so verdicts are not meaningful. The interface and "
        "explanations are fully functional and will use the real model "
        "automatically once it's saved.",
        icon="⚠️",
    )

email_text = st.text_area("Email text", height=220, placeholder="Paste the email here...")

if st.button("Analyze", type="primary"):
    if not email_text.strip():
        st.info("Please paste some email text first.")
    else:
        label, proba = classify(model, vectorizer, email_text)

        if label == 1:
            st.error(f"### ⚠️ Phishing  —  {proba:.0%} confidence")
        else:
            st.success(f"### ✅ Legitimate  —  {1 - proba:.0%} confidence")

        st.progress(proba)

        st.subheader("Why?")
        reasons = explain_prediction(model, vectorizer, email_text)
        for reason in reasons:
            st.markdown(f"- {reason}")

st.caption(
    "Built for CSE cybersecurity + AI course project. "
    "Verdicts are advisory — always use judgment with real email."
)
