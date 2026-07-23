"""
Feature engineering for the phishing detector  (Person 2 / Issue 2).

Produces the combined feature matrix that Person 3 trains on:
    X = [ TF-IDF features | 3 custom security features ]

Public functions (match the integration contract exactly):
    build_tfidf_vectorizer()  -> in preprocessing.py
    has_suspicious_link(text) -> int
    urgency_word_count(text)  -> int
    sender_mismatch(name, addr) -> int
    build_feature_matrix(df)  -> (X, y)

Extra helpers used by Person 4 (explanation + app), safe to ignore otherwise:
    extract_custom_features(text) -> [int, int, int]
    featurize_email(text, vectorizer) -> sparse row
    parse_sender(text) -> (display_name, sender_address)
"""

import os
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack

from src.preprocessing import build_tfidf_vectorizer

# Where build_feature_matrix saves the fitted vectorizer so Person 4 can reuse
# the *exact* same vectorizer at inference time. (Small addition to the contract
# — see the note at the bottom of this file.)
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

# ---------------------------------------------------------------------------
# Custom security feature #1 — suspicious links
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r'https?://[^\s<>"\')]+', re.IGNORECASE)
_IP_URL_RE = re.compile(r'https?://(?:\d{1,3}\.){3}\d{1,3}', re.IGNORECASE)
_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
}


def has_suspicious_link(email_text: str) -> int:
    """Return 1 if the email contains a suspicious link, else 0.

    Flags any of: a raw IP-address URL, a known URL shortener, an '@' inside a
    URL (a classic obfuscation trick), or a plain http:// (non-HTTPS) link.
    """
    text = str(email_text)
    urls = _URL_RE.findall(text)
    if not urls:
        return 0
    for url in urls:
        if _IP_URL_RE.match(url):
            return 1
        if "@" in url:                       # e.g. http://real.com@evil.com
            return 1
        if url.lower().startswith("http://"):  # non-HTTPS
            return 1
        # domain = text between :// and the next / , then take host
        host = url.split("://", 1)[1].split("/", 1)[0].lower()
        if any(host == s or host.endswith("." + s) for s in _SHORTENERS):
            return 1
    return 0


# ---------------------------------------------------------------------------
# Custom security feature #2 — urgency / pressure language
# ---------------------------------------------------------------------------
_URGENCY_PHRASES = [
    "urgent", "immediately", "act now", "right away", "as soon as possible",
    "asap", "verify your account", "verify now", "confirm your account",
    "account suspended", "account will be suspended", "suspended",
    "click here", "update your", "confirm your password", "password",
    "expire", "expires", "within 24 hours", "final notice", "last warning",
    "limited time", "your account will be", "unauthorized", "security alert",
    "action required", "failure to", "avoid suspension",
]


def urgency_word_count(email_text: str) -> int:
    """Return the number of urgency/pressure phrases found in the email."""
    text = str(email_text).lower()
    return sum(text.count(phrase) for phrase in _URGENCY_PHRASES)


# ---------------------------------------------------------------------------
# Custom security feature #3 — sender display-name vs address mismatch
# ---------------------------------------------------------------------------
_FREEMAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com"}
_FROM_RE = re.compile(
    r'from:\s*(?:"?([^"<\n]*?)"?\s*)?<?([\w.\-+]+@[\w.\-]+)>?',
    re.IGNORECASE,
)


def parse_sender(email_text: str):
    """Best-effort extraction of (display_name, sender_address) from raw text.

    Real emails often include a 'From:' header. If we can find one we return the
    parts; otherwise we return ('', '') and the mismatch feature stays 0. This
    keeps build_feature_matrix working even when Person 1's CSV is body-only.
    """
    m = _FROM_RE.search(str(email_text))
    if not m:
        return "", ""
    return (m.group(1) or "").strip(), (m.group(2) or "").strip()


def sender_mismatch(display_name: str, sender_address: str) -> int:
    """Return 1 if the display name looks inconsistent with the address, else 0.

    Simple heuristic (contract says a stub is fine):
    - No address to check  -> 0
    - A meaningful word from the display name (e.g. a brand) that does NOT appear
      anywhere in the email address -> mismatch
    - Display name claims a brand but the address is a free webmail account
      (paypal.com support really wouldn't email you from gmail) -> mismatch
    """
    name = (display_name or "").lower()
    addr = (sender_address or "").lower()
    if not addr or "@" not in addr:
        return 0

    local, _, domain = addr.partition("@")
    haystack = local + " " + domain

    name_tokens = [t for t in re.findall(r"[a-z]+", name) if len(t) > 2]
    if not name_tokens:
        return 0

    # If none of the display-name words show up in the address at all -> mismatch.
    if not any(tok in haystack for tok in name_tokens):
        return 1

    # A branded display name sent from free webmail is suspicious.
    if domain in _FREEMAIL and any(len(tok) > 3 for tok in name_tokens):
        return 1

    return 0


# ---------------------------------------------------------------------------
# Combine everything
# ---------------------------------------------------------------------------
def extract_custom_features(email_text: str):
    """Return the 3 custom features for one email, in a fixed order."""
    name, addr = parse_sender(email_text)
    return [
        has_suspicious_link(email_text),
        urgency_word_count(email_text),
        sender_mismatch(name, addr),
    ]


# Human-readable names for the 3 custom columns (used by the explainer).
CUSTOM_FEATURE_NAMES = ["[suspicious_link]", "[urgency_count]", "[sender_mismatch]"]


def build_feature_matrix(df, vectorizer=None, fit=True):
    """Turn a DataFrame of emails into (X, y).

    Input:  df with columns 'text' and 'label'  (Person 1's format).
    Output: (X, y) where
        X = sparse matrix [ TF-IDF | 3 custom features ], shape (n_samples, n_feats)
        y = df['label'].values

    Called as `build_feature_matrix(df)` per the contract. The two extra optional
    args let Person 4 reuse an already-fitted vectorizer at inference time.
    When fit=True the fitted vectorizer is saved to models/ so it can be loaded
    later alongside the trained model.
    """
    texts = df["text"].astype(str).tolist()

    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()

    if fit:
        tfidf = vectorizer.fit_transform(texts)
        _save_vectorizer(vectorizer)
    else:
        tfidf = vectorizer.transform(texts)

    custom = np.array([extract_custom_features(t) for t in texts], dtype=float)
    X = hstack([tfidf, csr_matrix(custom)]).tocsr()
    y = df["label"].values
    return X, y


def featurize_email(email_text: str, vectorizer):
    """Build the feature row for a SINGLE email using an already-fitted vectorizer.

    Column order matches build_feature_matrix exactly, so the trained model sees
    features in the same positions it was trained on. Used by Person 4.
    """
    tfidf = vectorizer.transform([str(email_text)])
    custom = np.array([extract_custom_features(email_text)], dtype=float)
    return hstack([tfidf, csr_matrix(custom)]).tocsr()


def feature_names(vectorizer):
    """Full ordered list of column names: TF-IDF tokens then the 3 custom ones."""
    return list(vectorizer.get_feature_names_out()) + CUSTOM_FEATURE_NAMES


def _save_vectorizer(vectorizer):
    try:
        import joblib
        os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
        joblib.dump(vectorizer, VECTORIZER_PATH)
    except Exception as exc:  # never let a save failure break training
        print(f"[features] warning: could not save vectorizer: {exc}")


# ---------------------------------------------------------------------------
# HEADS-UP FOR THE TEAM (one small addition to the contract):
# The contract only mentions saving the trained MODEL. But to classify a brand
# new email later, Person 4 also needs the SAME fitted TF-IDF vectorizer.
# So build_feature_matrix() now saves it to models/tfidf_vectorizer.joblib.
# Person 3: nothing changes for you — still call build_feature_matrix(df) and
# still get (X, y). The vectorizer just gets written to disk as a side effect.
# ---------------------------------------------------------------------------
