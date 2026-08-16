from __future__ import annotations

import re
import unicodedata
from typing import List

ARABIC_RE = re.compile(r"[^\w\u0600-\u06FF]+", flags=re.UNICODE)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for retrieval purposes.

    - NFKC normalization
    - lowercasing
    - remove tatweel
    - normalize alef variants to ا
    - normalize final ى to ي
    - remove diacritics
    - remove non-Arabic/word characters (keeps Arabic range and word chars)
    - collapse whitespace
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("ـ", "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    # remove Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # replace non-word / non-Arabic-range with space
    text = ARABIC_RE.sub(" ", text)
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    return normalize_arabic(text).split()


__all__ = ["normalize_arabic", "tokenize"]
