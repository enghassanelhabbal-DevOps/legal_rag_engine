from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from src.legal_ai.core.models import PipelineConfig, RuntimeConfig
from src.legal_ai.core.config import load_json, set_seed
from src.legal_ai.services.query_service import QueryService
from src.legal_ai.generation.manager import LLMManager
from src.legal_ai.evidence import build_grounded_context, select_evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("legal_rag_app")


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal RAG + Qwen3 Transformers, tuned for Quadro M2200")
    parser.add_argument("--documents", default="legal_documents.json")
    parser.add_argument("--artifact-dir", default="artifacts_m2200")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--dense-batch-size", type=int, default=4)
    parser.add_argument("--rerank-batch-size", type=int, default=1)
    parser.add_argument("--retrieval-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--llm-max-input-tokens", type=int, default=4096)
    parser.add_argument("--llm-max-new-tokens", type=int, default=320)
    args = parser.parse_args()

    set_seed()

    documents = load_json(Path(args.documents))
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    runtime = RuntimeConfig(
        device=args.retrieval_device,
        precision=args.precision,
        dense_batch_size=args.dense_batch_size,
        rerank_batch_size=args.rerank_batch_size,
        max_seq_length=384,
        compile_reranker=False,
        enable_tf32=False,
    )

    pipeline_cfg = PipelineConfig(
        dense_candidates=30,
        bm25_candidates=10,
        rerank_candidates=40,
        final_k=max(args.top_k, 5),
        rerank_max_chars=3500,
        alpha=0.75,
        max_context_chars=14000,
    )

    LOGGER.info("Preparing retrieval pipeline...")
    rag = QueryService(documents, runtime, pipeline_cfg, artifact_dir, load_reranker=not args.no_reranker)

    # Important M2200 policy:
    # Keep the LLM as the final GPU-heavy model. Retrieval components can be
    # unloaded immediately before Qwen is loaded to reduce VRAM pressure.
    retrieval = rag.retrieve(args.query, top_k=args.top_k)

    evidence = select_evidence(retrieval["results"], max_chars=pipeline_cfg.max_context_chars)
    context = build_grounded_context(evidence, max_chars=pipeline_cfg.max_context_chars)

    LOGGER.info("Unloading retrieval Transformer models before Qwen load...")
    rag.close()

    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    llm = LLMManager({
        "max_input_tokens": args.llm_max_input_tokens,
        "max_new_tokens": args.llm_max_new_tokens,
        "m2200_gpu_memory": "3.2GiB",
        "cpu_memory_budget": "12GiB",
        "offload_folder": str(artifact_dir / "qwen_offload"),
    })
    llm.load()

    t0 = time.perf_counter()
    answer = llm.generate(args.query, context)
    generation_ms = (time.perf_counter() - t0) * 1000

    output = {
        "query": args.query,
        "answer": answer,
        "sources": retrieval["results"],
        "context": context,
        "retrieval_latency_ms": retrieval["latency_ms"],
        "generation_ms": generation_ms,
        "runtime": rag.runtime_info,
        "llm": llm.info(),
    }

    out_file = artifact_dir / "last_answer.json"
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ANSWER ===\n")
    print(answer)
    print("\n=== SOURCES ===\n")
    for i, source in enumerate(retrieval["results"], 1):
        title = (source.get("metadata") or {}).get("title", "مصدر قانوني")
        print(f"[{i}] {title} | id={source.get('id')}")

    print(f"\nRetrieval latency: {retrieval['latency_ms']['end_to_end_ms']:.1f} ms")
    print(f"Generation latency: {generation_ms:.1f} ms")
    print(f"LLM model: {llm.info()['actual_model']}")


if __name__ == "__main__":
    main()
