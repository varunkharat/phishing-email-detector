"""Highlight the phrases/links that most influenced a phishing prediction."""


def explain_prediction(model, vectorizer, email_text: str) -> list[str]:
    """Return the top contributing terms/phrases for the model's verdict on this email."""
    raise NotImplementedError
