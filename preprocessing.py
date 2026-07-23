"""
Text preprocessing for the phishing detector.

NOTE ON OWNERSHIP:
- `clean_text()` belongs to Person 1 (Issue 1). If it's still a placeholder,
  that's fine — `build_tfidf_vectorizer()` does its own lowercasing/tokenizing,
  so the feature pipeline works whether or not clean_text has been filled in yet.
- `build_tfidf_vectorizer()` below is Person 2's (Issue 2).

If Person 1 has already written clean_text() in this file, DO NOT overwrite it —
just paste `build_tfidf_vectorizer` in alongside it.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """Placeholder — owned by Person 1 (Issue 1).

    Kept here only so imports don't break. The TF-IDF vectorizer normalizes
    text on its own, so downstream code does not depend on this being finished.
    """
    return text if isinstance(text, str) else str(text)


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Return a ready-to-fit TfidfVectorizer.

    The caller (build_feature_matrix) will call .fit_transform() on it.

    Choices explained:
    - stop_words="english": drops filler words ("the", "and") so real signal wins.
    - ngram_range=(1, 2): captures single words AND two-word phrases like
      "verify account" or "click here", which matter a lot for phishing.
    - sublinear_tf=True: dampens the effect of a word repeated many times.
    - min_df=2: ignore words that appear in only one email (usually noise).
    """
    return TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
    )
