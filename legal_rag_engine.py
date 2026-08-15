from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import faiss
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from legal_ai.knowledge import EmbeddingCache, KnowledgeVersion


SEED = 42
DENSE_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

ARABIC_RE = re.compile(r"[^\w\u0600-\u06FF]+", flags=re.UNICODE)

LOGGER = logging.getLogger("legal_rag")


# -----------------------------------------------------------------------------
# Runtime / hardware
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    gpu_id: int = 0
    precision: str = "auto"  # auto | fp32 | fp16 | bf16
    dense_batch_size: int = 32
    rerank_batch_size: int = 32
    num_threads: int = 0
    enable_tf32: bool = True
    compile_reranker: bool = False
    max_seq_length: int = 1024


def configure_runtime(cfg: RuntimeConfig) -> str:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)

    if cfg.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.device == "cpu":
        device = "cpu"
    elif cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA GPU.")
        device = f"cuda:{cfg.gpu_id}"
    else:
        device = f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu"

    LOGGER.info("Device: %s", device)
    if device.startswith("cuda"):
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(cfg.gpu_id))
        LOGGER.info("CUDA: %s", torch.version.cuda)

    return device


def choose_dtype(cfg: RuntimeConfig, device: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    if cfg.precision == "fp32":
        return torch.float32
    if cfg.precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        LOGGER.warning("BF16 is not supported; falling back to FP16.")
        return torch.float16
    if cfg.precision == "fp16":
        return torch.float16
    # auto
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Retrieval itself is deterministic. Training is not part of this pipeline.
    torch.use_deterministic_algorithms(False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


from legal_ai.normalization import normalize_arabic, tokenize


def metadata_text(doc: dict) -> str:
    metadata = doc.get("metadata") or {}
    title = clean_text(metadata.get("title", ""))
    content = clean_text(doc.get("content", ""))
    if title:
        return f"العنوان: {title}\nالنص القانوني: {content}"
    return content


def lexical_text(doc: dict) -> str:
    metadata = doc.get("metadata") or {}
    title = clean_text(metadata.get("title", ""))
    content = clean_text(doc.get("content", ""))
    # Small title boost for exact article/law terms.
    return f"{title} {title} {content}".strip()


def validate_documents(documents: list[dict]) -> None:
    if not documents:
        raise ValueError("No documents found.")
    ids = [str(d.get("id", "")) for d in documents]
    if any(not x for x in ids):
        raise ValueError("Every document needs a non-empty id.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document ids found.")
    for i, doc in enumerate(documents):
        if not isinstance(doc.get("content"), str) or not doc["content"].strip():
            raise ValueError(f"Empty content at row {i} / id={doc.get('id')}")
        if not isinstance(doc.get("metadata"), dict):
            raise ValueError(f"Invalid metadata at row {i} / id={doc.get('id')}")


# -----------------------------------------------------------------------------
# BM25
# -----------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus_size = len(corpus_tokens)
        self.doc_len = np.asarray([len(x) for x in corpus_tokens], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if self.corpus_size else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for tokens in corpus_tokens:
            freqs = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            for token in freqs:
                df[token] = df.get(token, 0) + 1
        self.idf = {
            term: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def top_n(self, query: str, n: int) -> list[tuple[int, float]]:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        qtf: dict[str, int] = {}
        for token in tokenize(query):
            qtf[token] = qtf.get(token, 0) + 1
        if not qtf or self.corpus_size == 0 or self.avgdl == 0:
            return []

        for term in qtf:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1.0) / denom)

        n = min(max(1, n), len(scores))
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [(int(i), float(scores[i])) for i in idx]


# -----------------------------------------------------------------------------
# Retrieval models
# -----------------------------------------------------------------------------
@dataclass
class RetrievalHit:
    id: str
    index: int
    content: str
    metadata: dict[str, Any]
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    reranker_score: float | None = None
    final_score: float | None = None
    final_rank: int | None = None
    sources: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DenseEncoder:
    def __init__(self, model_name: str, device: str, dtype: torch.dtype, max_seq_length: int):
        LOGGER.info("Loading dense model: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        if device.startswith("cuda") and dtype == torch.float16:
            self.model.half()
        self.device = device
        self.dtype = dtype

    def encode_documents(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        autocast = torch.autocast(device_type="cuda", dtype=self.dtype) if self.device.startswith("cuda") and self.dtype == torch.bfloat16 else nullcontext()
        with torch.inference_mode(), autocast:
            embeddings = self.model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        autocast = torch.autocast(device_type="cuda", dtype=self.dtype) if self.device.startswith("cuda") and self.dtype == torch.bfloat16 else nullcontext()
        with torch.inference_mode(), autocast:
            embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(embedding, dtype=np.float32)


class DenseIndex:
    def __init__(self, embeddings: np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(query_embedding, min(k, self.index.ntotal))

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @staticmethod
    def load(path: Path) -> "DenseIndex":
        obj = DenseIndex.__new__(DenseIndex)
        obj.index = faiss.read_index(str(path))
        return obj


class Reranker:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: torch.dtype,
        max_seq_length: int,
        compile_model: bool = False,
    ):
        LOGGER.info("Loading reranker: %s", model_name)
        model_kwargs = {}
        if device.startswith("cuda"):
            model_kwargs["torch_dtype"] = dtype
        self.model = CrossEncoder(model_name, device=device, max_length=max_seq_length, model_kwargs=model_kwargs)
        self.device = device

        if compile_model and device.startswith("cuda") and hasattr(self.model, "compile"):
            try:
                self.model.compile(dynamic=True)
                LOGGER.info("CrossEncoder torch.compile enabled.")
            except Exception as exc:
                LOGGER.warning("torch.compile unavailable for reranker: %s", exc)

    def score(self, query: str, candidates: Sequence[RetrievalHit], batch_size: int, max_chars: int) -> np.ndarray:
        pairs = []
        for c in candidates:
            text = metadata_text({"content": c.content, "metadata": c.metadata})[:max_chars]
            pairs.append([query, text])

        with torch.inference_mode():
            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                apply_softmax=False,
            )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    dense_candidates: int = 30
    bm25_candidates: int = 10
    rerank_candidates: int = 40
    final_k: int = 10
    rerank_max_chars: int = 4500
    alpha: float = 0.75
    max_context_chars: int = 32000


class HybridRetriever:
    """Reusable retrieval core; LLM integration is intentionally separated."""

    def __init__(
        self,
        documents: list[dict],
        encoder: DenseEncoder,
        dense_index: DenseIndex,
        bm25: BM25,
        reranker: Reranker | None,
        config: PipelineConfig,
    ):
        self.documents = documents
        self.encoder = encoder
        self.dense_index = dense_index
        self.bm25 = bm25
        self.reranker = reranker
        self.config = config

    def dense_search(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        k = k or self.config.dense_candidates
        q = self.encoder.encode_query(query)
        scores, indices = self.dense_index.search(q, k)
        results: list[RetrievalHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            d = self.documents[int(idx)]
            results.append(
                RetrievalHit(
                    id=str(d["id"]),
                    index=int(idx),
                    content=d["content"],
                    metadata=d.get("metadata", {}),
                    dense_score=float(score),
                    dense_rank=rank,
                    sources=["dense"],
                )
            )
        return results

    def bm25_search(self, query: str, k: int | None = None) -> list[RetrievalHit]:
        k = k or self.config.bm25_candidates
        results: list[RetrievalHit] = []
        for rank, (idx, score) in enumerate(self.bm25.top_n(query, k), start=1):
            d = self.documents[idx]
            results.append(
                RetrievalHit(
                    id=str(d["id"]),
                    index=idx,
                    content=d["content"],
                    metadata=d.get("metadata", {}),
                    bm25_score=score,
                    bm25_rank=rank,
                    sources=["bm25"],
                )
            )
        return results

    @staticmethod
    def dense_preserving_union(dense: list[RetrievalHit], bm25: list[RetrievalHit], max_candidates: int) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        for hit in dense:
            merged[hit.id] = RetrievalHit(**{**asdict(hit), "sources": ["dense"]})
        for hit in bm25:
            if hit.id in merged:
                x = merged[hit.id]
                x.bm25_score = hit.bm25_score
                x.bm25_rank = hit.bm25_rank
                x.sources = sorted(set((x.sources or []) + ["bm25"]))
            else:
                merged[hit.id] = RetrievalHit(**{**asdict(hit), "sources": ["bm25"]})

        # Dense results are guaranteed to stay at the head of the pool.
        ordered = [merged[x.id] for x in dense]
        ordered += [merged[x.id] for x in bm25 if x.id not in {d.id for d in dense}]
        return ordered[:max_candidates]

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo < 1e-8:
            return np.full_like(values, 0.5, dtype=np.float32)
        return (values - lo) / (hi - lo)

    def rerank(self, query: str, pool: list[RetrievalHit]) -> list[RetrievalHit]:
        if self.reranker is None or not pool:
            return pool
        scores = self.reranker.score(query, pool, self.config.rerank_candidates, self.config.rerank_max_chars)
        for hit, score in zip(pool, scores):
            hit.reranker_score = float(score)
        return pool

    def fuse(self, hits: list[RetrievalHit], alpha: float | None = None) -> list[RetrievalHit]:
        if not hits:
            return []
        alpha = self.config.alpha if alpha is None else float(alpha)
        rr = self._minmax(np.asarray([x.reranker_score or 0.0 for x in hits], dtype=np.float32))
        max_rank = max(1, len(hits) - 1)
        dense_prior = np.asarray(
            [
                1.0 - (x.dense_rank - 1) / max_rank if x.dense_rank is not None else 0.0
                for x in hits
            ],
            dtype=np.float32,
        )
        for hit, rr_score, prior in zip(hits, rr, dense_prior):
            hit.final_score = float(alpha * prior + (1.0 - alpha) * rr_score)
        hits.sort(key=lambda x: (-float(x.final_score or -1e9), x.id))
        for rank, hit in enumerate(hits, start=1):
            hit.final_rank = rank
        return hits

    def retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        top_k = top_k or self.config.final_k
        t0 = time.perf_counter()
        dense = self.dense_search(query)
        t1 = time.perf_counter()
        bm25 = self.bm25_search(query)
        t2 = time.perf_counter()
        pool = self.dense_preserving_union(dense, bm25, self.config.rerank_candidates)
        t3 = time.perf_counter()

        if self.reranker:
            pool = self.rerank(query, pool)
            t4 = time.perf_counter()
            final = self.fuse(pool)[:top_k]
        else:
            t4 = t3
            final = pool[:top_k]

        latency = {
            "dense_ms": (t1 - t0) * 1000,
            "bm25_ms": (t2 - t1) * 1000,
            "candidate_ms": (t3 - t2) * 1000,
            "rerank_ms": (t4 - t3) * 1000,
            "end_to_end_ms": (time.perf_counter() - t0) * 1000,
        }
        return {
            "query": query,
            "results": [x.as_dict() for x in final],
            "dense": [x.as_dict() for x in dense],
            "bm25": [x.as_dict() for x in bm25],
            "candidate_pool": [x.as_dict() for x in pool],
            "latency_ms": latency,
        }


# -----------------------------------------------------------------------------
# LLM-ready interface
# -----------------------------------------------------------------------------
class LLMBackend(Protocol):
    def generate(self, query: str, context: str) -> str: ...


def build_grounded_context(results: Sequence[dict], max_chars: int = 32000) -> str:
    blocks = []
    current = 0
    for i, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        title = clean_text(metadata.get("title", "")) or "مصدر قانوني"
        content = clean_text(item.get("content", ""))
        block = f"[SOURCE {i}]\nالعنوان: {title}\nالنص: {content}\nمعرّف المصدر: {item.get('id')}"
        if current + len(block) > max_chars:
            break
        blocks.append(block)
        current += len(block)
    return "\n\n".join(blocks)


class RAGService:
    def __init__(self, retriever: HybridRetriever, llm: LLMBackend | None = None):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query: str, top_k: int = 5) -> dict[str, Any]:
        retrieval = self.retriever.retrieve(query, top_k=top_k)
        context = build_grounded_context(retrieval["results"], self.retriever.config.max_context_chars)
        if self.llm is None:
            return {"query": query, "context": context, "sources": retrieval["results"], "retrieval": retrieval}
        answer = self.llm.generate(query, context)
        return {
            "query": query,
            "answer": answer,
            "context": context,
            "sources": retrieval["results"],
            "retrieval": retrieval,
        }


# -----------------------------------------------------------------------------
# Persistence / build
# -----------------------------------------------------------------------------
def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_index(
    documents: list[dict],
    encoder: DenseEncoder,
    bm25: BM25,
    batch_size: int,
    out_dir: Path,
) -> DenseIndex:
    embeddings_path = out_dir / "dense_embeddings.npy"
    index_path = out_dir / "dense_faiss.index"
    cache_path = out_dir / "embeddings_cache.json"
    cache = EmbeddingCache(cache_path)

    if embeddings_path.exists() and index_path.exists():
        LOGGER.info("Loading cached dense embeddings and FAISS index.")
        embeddings = np.load(embeddings_path, mmap_mode="r")
        index = DenseIndex.load(index_path)
        if index.index.ntotal != len(documents):
            raise ValueError("Cached index size does not match current document count.")
        return index

    texts = [metadata_text(d) for d in documents]
    doc_hashes = [sha256_text(t) for t in texts]

    # Pre-fill embeddings from cache where possible
    embeddings_list = [None] * len(documents)
    to_encode_indices: list[int] = []
    to_encode_texts: list[str] = []
    for i, h in enumerate(doc_hashes):
        emb = cache.get(h)
        if emb is not None:
            embeddings_list[i] = np.asarray(emb, dtype=np.float32)
        else:
            to_encode_indices.append(i)
            to_encode_texts.append(texts[i])

    # Encode missing embeddings in batches
    if to_encode_texts:
        LOGGER.info("Encoding %d missing embeddings (batch_size=%d)", len(to_encode_texts), batch_size)
        computed = encoder.encode_documents(to_encode_texts, batch_size=batch_size)
        # computed is numpy array shaped (N, dim)
        for idx, emb in zip(to_encode_indices, computed):
            embeddings_list[idx] = np.asarray(emb, dtype=np.float32)
            cache.set(doc_hashes[idx], emb.tolist())
        # persist incremental cache
        cache.persist()

    # Stack into embeddings array
    embeddings = np.vstack(embeddings_list).astype(np.float32)

    index = DenseIndex(embeddings)
    np.save(embeddings_path, embeddings)
    index.save(index_path)
    return index


def prepare_pipeline(
    documents: list[dict],
    runtime: RuntimeConfig,
    pipeline_cfg: PipelineConfig,
    out_dir: Path,
    load_reranker: bool = True,
) -> tuple[HybridRetriever, dict[str, Any]]:
    device = configure_runtime(runtime)
    dtype = choose_dtype(runtime, device)
    LOGGER.info("Inference dtype: %s", dtype)

    encoder = DenseEncoder(DENSE_MODEL_NAME, device, dtype, runtime.max_seq_length)
    bm25 = BM25([tokenize(lexical_text(d)) for d in documents])
    index = build_index(documents, encoder, bm25, runtime.dense_batch_size, out_dir)

    reranker = None
    if load_reranker:
        reranker = Reranker(
            RERANKER_MODEL_NAME,
            device,
            dtype,
            runtime.max_seq_length,
            compile_model=runtime.compile_reranker,
        )

    retriever = HybridRetriever(documents, encoder, index, bm25, reranker, pipeline_cfg)
    info = {
        "device": device,
        "dtype": str(dtype),
        "gpu_name": torch.cuda.get_device_name(runtime.gpu_id) if device.startswith("cuda") else None,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
    }
    save_json(info, out_dir / "runtime_info.json")

    # Knowledge versioning: create a simple reproducible dataset hash and save manifest
    try:
        dataset_hash_h = hashlib.sha256()
        for d in documents:
            dataset_hash_h.update(str(d.get("id")).encode("utf-8"))
            dataset_hash_h.update(str(d.get("content", "")).encode("utf-8"))
        dataset_hash = dataset_hash_h.hexdigest()
        kv = KnowledgeVersion(
            version_id=f"v{int(time.time())}",
            dataset_hash=dataset_hash,
            document_count=len(documents),
            embedding_model=DENSE_MODEL_NAME,
            index_version="faiss-flat-ip",
        )
        kv.save(out_dir / "knowledge_version.json")
    except Exception as exc:
        LOGGER.warning("Could not write knowledge version manifest: %s", exc)

    return retriever, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable GPU-accelerated Legal RAG retrieval core")
    parser.add_argument("--documents", default="legal_documents (1).json")
    parser.add_argument("--output", default="artifacts_portable")
    parser.add_argument("--query", default="ما شروط القبض على المتهم في حالة التلبس؟")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-candidates", type=int, default=30)
    parser.add_argument("--bm25-candidates", type=int, default=10)
    parser.add_argument("--rerank-candidates", type=int, default=40)
    parser.add_argument("--rerank-max-chars", type=int, default=4500)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--compile-reranker", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed()

    docs_path = Path(args.documents)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    documents = load_json(docs_path)
    validate_documents(documents)

    runtime = RuntimeConfig(
        device=args.device,
        gpu_id=args.gpu_id,
        precision=args.precision,
        dense_batch_size=args.dense_batch_size,
        rerank_batch_size=args.rerank_batch_size,
        max_seq_length=args.max_seq_length,
        compile_reranker=args.compile_reranker,
    )
    pipeline_cfg = PipelineConfig(
        dense_candidates=args.dense_candidates,
        bm25_candidates=args.bm25_candidates,
        rerank_candidates=args.rerank_candidates,
        final_k=args.top_k,
        rerank_max_chars=args.rerank_max_chars,
        alpha=args.alpha,
    )

    retriever, runtime_info = prepare_pipeline(
        documents,
        runtime,
        pipeline_cfg,
        out_dir,
        load_reranker=not args.no_reranker,
    )

    save_json(
        {
            "seed": SEED,
            "documents_sha256": sha256_file(docs_path),
            "runtime": asdict(runtime),
            "pipeline": asdict(pipeline_cfg),
            "models": {
                "dense": DENSE_MODEL_NAME,
                "reranker": None if args.no_reranker else RERANKER_MODEL_NAME,
            },
            "runtime_info": runtime_info,
        },
        out_dir / "run_config.json",
    )

    result = retriever.retrieve(args.query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Release GPU memory explicitly for notebook/service reloading scenarios.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
