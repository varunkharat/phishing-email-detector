"""Hand-crafted, security-specific feature extraction."""


def has_suspicious_link(email_text: str) -> bool:
    """Detect mismatched/obfuscated URLs, IP-based links, known shortener abuse, etc."""
    raise NotImplementedError


def urgency_word_count(email_text: str) -> int:
    """Count occurrences of urgency phrases, e.g. 'verify now', 'account suspended'."""
    raise NotImplementedError


def sender_mismatch(display_name: str, sender_address: str) -> bool:
    """Flag when the display name and the actual sending domain don't align."""
    raise NotImplementedError
