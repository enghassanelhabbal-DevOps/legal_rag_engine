"""versioning.py — Knowledge versioning, manifest, and rollback utilities.

Migrated from legal_ai/knowledge.py (KnowledgeVersion + rollback helpers).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class KnowledgeVersion:
    """Immutable snapshot descriptor for a knowledge base build.

    Saved alongside each re-index so that retrieval baselines can be reproduced
    and rolled back (ARCHITECTURE_CONTRACT.md §Protected retrieval baseline).
    """

    version_id: str
    dataset_hash: str           # SHA-256 of the canonical documents JSON
    document_count: int
    embedding_model: str        # e.g. "BAAI/bge-m3"
    embedding_model_hash: str | None = None   # model weights hash
    reranker_model: str | None = None
    index_type: str = "faiss-flat-ip"
    embedding_cache_path: str | None = None
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            object.__setattr__(self, "created_at", time.time())

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: Path) -> KnowledgeVersion:
        data = json.loads(path.read_text(encoding="utf-8"))
        return KnowledgeVersion(**data)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def list_versions(dir_path: Path) -> list[Path]:
    """Return all knowledge-version manifest files sorted by name."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    return sorted(dir_path.glob("knowledge_version*.json"))


def rollback_to(version_path: Path, artifacts_dir: Path) -> None:
    """Attempt a conservative rollback using a saved KnowledgeVersion manifest.

    Only copies known artifact files that physically exist next to the manifest.
    No destructive operations are performed automatically.
    """
    version = KnowledgeVersion.load(version_path)
    src_dir = version_path.parent
    candidates = {
        "embeddings": src_dir / "dense_embeddings.npy",
        "index": src_dir / "dense_faiss.index",
        "emb_cache": (
            Path(version.embedding_cache_path)
            if version.embedding_cache_path
            else src_dir / "embeddings_cache.json"
        ),
    }
    for _name, src in candidates.items():
        if src and src.exists():
            dst = Path(artifacts_dir) / src.name
            dst.write_bytes(src.read_bytes())


__all__ = ["KnowledgeVersion", "list_versions", "rollback_to"]
