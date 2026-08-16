from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class KnowledgeVersion:
    version_id: str
    dataset_hash: str
    document_count: int
    embedding_model: str
    embedding_model_hash: str | None = None
    reranker_model: str | None = None
    index_type: str = "faiss-flat-ip"
    embedding_cache_path: str | None = None
    created_at: float = time.time()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> KnowledgeVersion:
        data = json.loads(path.read_text(encoding="utf-8"))
        return KnowledgeVersion(**data)


# Embedding cache implementations ------------------------------------------------
class EmbeddingCacheJSON:
    """JSON-backed embedding cache keyed by document hash."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, list[float]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get(self, doc_hash: str):
        return self._data.get(doc_hash)

    def set(self, doc_hash: str, embedding: list[float]) -> None:
        self._data[doc_hash] = embedding

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")


class EmbeddingCacheLMDB:
    """LMDB-backed embedding cache for better performance on larger datasets.

    This optional implementation uses the 'lmdb' package. If lmdb is not
    installed, the factory will fall back to EmbeddingCacheJSON.
    """

    def __init__(self, path: Path, map_size: int = 1 << 30):
        import struct

        import lmdb

        self.path = path
        self._env = lmdb.open(str(path), map_size=map_size, subdir=False, lock=True)
        self._struct = struct

    def get(self, doc_hash: str):
        with self._env.begin() as txn:
            v = txn.get(doc_hash.encode("utf-8"))
            if v is None:
                return None
            # stored as JSON bytes
            return json.loads(v.decode("utf-8"))

    def set(self, doc_hash: str, embedding: list[float]) -> None:
        with self._env.begin(write=True) as txn:
            txn.put(doc_hash.encode("utf-8"), json.dumps(embedding, ensure_ascii=False).encode("utf-8"))

    def persist(self) -> None:
        # LMDB commits on put; nothing special to do.
        return

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


class EmbeddingCache:
    """Factory for embedding caches. Choose LMDB if available, otherwise JSON."""

    def __init__(self, path: Path):
        self.path = path
        # prefer LMDB if available
        try:

            # LMDB stores a single file; ensure parent exists
            path.parent.mkdir(parents=True, exist_ok=True)
            self._impl = EmbeddingCacheLMDB(path.with_suffix("").with_name(path.name + ".lmdb"))
        except Exception:
            self._impl = EmbeddingCacheJSON(path)

    def get(self, doc_hash: str):
        return self._impl.get(doc_hash)

    def set(self, doc_hash: str, embedding: list[float]) -> None:
        return self._impl.set(doc_hash, embedding)

    def persist(self) -> None:
        return self._impl.persist()

    def close(self) -> None:
        if hasattr(self._impl, "close"):
            try:
                self._impl.close()
            except Exception:
                pass


# Knowledge rollback utilities ---------------------------------------------------

def list_versions(dir_path: Path) -> list[Path]:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob("knowledge_version*.json")])


def rollback_to(version_path: Path, artifacts_dir: Path) -> None:
    """Attempt to rollback embeddings/index/manifest using a saved KnowledgeVersion file.

    This is intentionally conservative: it only copies known artifact paths if they exist in the
    same directory as the version manifest. It does not attempt destructive operations automatically.
    """
    version = KnowledgeVersion.load(version_path)
    src_dir = version_path.parent
    # Copy known artifacts if they exist in src_dir
    candidates = {
        "embeddings": src_dir / "dense_embeddings.npy",
        "index": src_dir / "dense_faiss.index",
        "emb_cache": Path(version.embedding_cache_path) if version.embedding_cache_path else (src_dir / "embeddings_cache.json"),
    }
    for name, src in candidates.items():
        if src and src.exists():
            dst = Path(artifacts_dir) / src.name
            # overwrite
            dst.write_bytes(src.read_bytes())


__all__ = ["KnowledgeVersion", "EmbeddingCache", "list_versions", "rollback_to"]
