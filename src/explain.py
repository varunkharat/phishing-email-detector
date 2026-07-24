"""
Explanation module for the phishing detector  (Person 4 / Issue 4).

Given a model, its vectorizer, and an email, return a short list of plain-English
reasons for the verdict. Works with:
  - real linear models (LogisticRegression, Naive Bayes) -> shows top words
  - RandomForest / anything else                          -> rule-based reasons
  - the FakeModel placeholder                             -> rule-based reasons
so the UI can be built and demoed today, before Person 3's model exists.
"""

import numpy as np

from src.features import (
    featurize_email,
    feature_names,
    has_suspicious_link,
    urgency_word_count,
    parse_sender,
    sender_mismatch,
)


def explain_prediction(model, vectorizer, email_text: str) -> list:
    """Return a list of strings: the reasons this email got its verdict.

    Combines two kinds of evidence:
      1. Rule-based signals (always available, model-agnostic, very readable).
      2. Model-driven top words, when the model exposes signed weights (coef_).
    """
    reasons = []

    # --- 1. Rule-based signals (from Person 2's features) -------------------
    if has_suspicious_link(email_text):
        reasons.append(
            "Contains a suspicious link (IP-address URL, shortener, '@' trick, "
            "or non-HTTPS)."
        )

    n_urgency = urgency_word_count(email_text)
    if n_urgency:
        reasons.append(
            f"Uses {n_urgency} urgency/pressure phrase(s) common in phishing "
            "(e.g. 'verify now', 'account suspended')."
        )

    name, addr = parse_sender(email_text)
    if addr and sender_mismatch(name, addr):
        reasons.append(
            "Sender display name doesn't match the actual sending address."
        )

    # --- 2. Model-driven top words -----------------------------------------
    # Try linear coefficients first (LogReg / Naive Bayes); if the model
    # doesn't expose those (e.g. RandomForest), fall back to SHAP so tree
    # models still get real per-email word explanations instead of only the
    # rule-based reasons above.
    try:
        names = feature_names(vectorizer) if vectorizer is not None else None
        row = (
            featurize_email(email_text, vectorizer).toarray()[0]
            if vectorizer is not None
            else None
        )
        top_words = None

        coef = _get_linear_coef(model)
        if coef is not None and row is not None:
            contribution = coef * row  # weight * value = push toward phishing
            order = np.argsort(contribution)[::-1]
            top_words = [
                names[i]
                for i in order[:6]
                if contribution[i] > 0 and not names[i].startswith("[")
            ][:4]

        if not top_words and row is not None:
            pass  # SHAP disabled — was hanging/crashing on the large RF model
            # top_words = _get_tree_shap_top_words(model, row, names)

        if top_words:
            reasons.append(
                "Wording that pushed this toward 'phishing': "
                + ", ".join(f"'{w}'" for w in top_words)
            )
    except Exception:
        # Explanation must never crash the app; rule-based reasons still stand.
        pass

    if not reasons:
        reasons.append(
            "No strong individual red flags found — verdict is based on the "
            "overall wording of the email."
        )

    return reasons


def _get_tree_shap_top_words(model, row, names):
    """Return top words pushing toward 'phishing' for tree models (e.g. RandomForest).

    Uses SHAP's TreeExplainer, which is fast for tree ensembles and doesn't
    need a background dataset. Returns None if the model isn't tree-based,
    shap isn't available, or anything goes wrong — caller treats None the
    same as "no model-driven words available" and keeps the rule-based
    reasons.
    """
    if not hasattr(model, "feature_importances_") or names is None:
        return None
    try:
        import shap
    except ImportError:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row.reshape(1, -1))

    # Different shap/sklearn version combos return either a list of per-class
    # arrays or a single array — handle both, always taking class 1 (phishing).
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = np.asarray(shap_values)[0]
        if values.ndim > 1:
            values = values[:, 1]

    order = np.argsort(values)[::-1]
    return [
        names[i] for i in order[:6]
        if values[i] > 0 and not names[i].startswith("[")
    ][:4]


def _get_linear_coef(model):
    """Return a 1-D coefficient vector for linear models, else None."""
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        return coef[0] if coef.ndim > 1 else coef
    # Naive Bayes: difference of class log-probs gives a usable signed weight.
    if hasattr(model, "feature_log_prob_"):
        logp = np.asarray(model.feature_log_prob_)
        if logp.shape[0] == 2:
            return logp[1] - logp[0]
    return None
