from __future__ import annotations

from typing import Protocol


class LLMBackend(Protocol):
    def generate(self, query: str, context: str) -> str: ...


class ExampleLocalLLM:
    """Adapter interface only. Replace with Transformers/vLLM/Ollama/etc."""

    def __init__(self, model):
        self.model = model

    def generate(self, query: str, context: str) -> str:
        prompt = f"""You are a legal information assistant.
Use ONLY the supplied legal context.
Cite the SOURCE number for every legal claim.
If the context is insufficient, say so explicitly.
Do not invent laws or articles.

QUESTION:
{query}

LEGAL CONTEXT:
{context}
"""
        # Implement your selected LLM backend here.
        # Return only the final grounded answer string.
        raise NotImplementedError("Connect your LLM backend here.")
