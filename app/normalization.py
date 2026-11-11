"""API-side Persian text normalization.

This intentionally lives outside the preprocessing notebooks. Keep its rules
aligned with the normalization used to prepare the training text.
"""

from __future__ import annotations

import html
import re
import unicodedata


ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ك": "ک"})
URL_PATTERN = re.compile(r"(?i)(?:https?://|www\.)\S+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
INVISIBLE_PATTERN = re.compile(r"[\u200b\ufeff\u200e\u200f\u202a-\u202e\u2066-\u2069]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_persian_text(value: str) -> str:
    """Normalize Persian text while preserving sentiment, numbers and emoji."""
    if not isinstance(value, str):
        raise TypeError("text must be a string")

    text = html.unescape(value).translate(ARABIC_TO_PERSIAN)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" <URL> ", text)
    text = INVISIBLE_PATTERN.sub("", text)
    return WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", text)).strip()

