"""Text cleaning and TF-IDF vectorization for email bodies."""

from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(raw_email: str) -> str:
    """Lowercase, strip HTML/headers, remove extra whitespace, etc."""
    raise NotImplementedError


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Return a configured TF-IDF vectorizer to fit on training email text."""
    raise NotImplementedError
