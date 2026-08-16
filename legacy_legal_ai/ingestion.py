from __future__ import annotations

import re
from typing import Any

ARTICLE_RE = re.compile(r"(مادة|المادة)\s*(?:رقم\s*)?(\d+)", flags=re.IGNORECASE)


def article_aware_chunk(text: str, max_chars: int = 2000) -> list[dict[str, Any]]:
    """Split a legislative text into article-aware chunks.

    Returns list of dicts: { 'article_id': str|None, 'text': str, 'chunk_index': int, 'chunk_count': int }

    Behavior:
    - Attempt to split by explicit article headings (مادة / المادة). If none found,
      fall back to simple windowed splits while keeping chunk sizes under max_chars.
    - If an article is larger than max_chars, split it into multiple chunks preserving parent article id.
    """
    # Find article boundaries
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        # fallback: simple slicing
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append({
                "article_id": None,
                "text": text[i : i + max_chars].strip(),
                "chunk_index": i // max_chars,
                "chunk_count": (len(text) + max_chars - 1) // max_chars,
            })
        return chunks

    # Build article spans
    spans = []
    for i, m in enumerate(matches):
        start = m.start()
        article_num = m.group(2)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((article_num, text[start:end].strip()))

    # Now chunk each article if needed
    out: list[dict[str, Any]] = []
    for article_id, body in spans:
        if len(body) <= max_chars:
            out.append({
                "article_id": article_id,
                "text": body,
                "chunk_index": 0,
                "chunk_count": 1,
            })
            continue
        # split into multiple chunks
        parts = [body[i : i + max_chars].strip() for i in range(0, len(body), max_chars)]
        for idx, part in enumerate(parts):
            out.append(
                {
                    "article_id": article_id,
                    "text": part,
                    "chunk_index": idx,
                    "chunk_count": len(parts),
                }
            )
    return out


def chunk_document(doc: dict[str, Any], max_chars: int = 2000) -> list[dict[str, Any]]:
    """Create chunks for a document dict with keys 'id', 'content', 'metadata'.
    Returns list of chunk dicts with provenance fields.
    """
    content = doc.get("content", "") or ""
    chunks = article_aware_chunk(content, max_chars=max_chars)
    out: list[dict[str, Any]] = []
    for c in chunks:
        chunk_meta = {
            "doc_id": doc.get("id"),
            "article_id": c.get("article_id"),
            "chunk_index": c.get("chunk_index"),
            "chunk_count": c.get("chunk_count"),
            "metadata": doc.get("metadata", {}),
            "text": c.get("text"),
        }
        out.append(chunk_meta)
    return out


__all__ = ["article_aware_chunk", "chunk_document"]
