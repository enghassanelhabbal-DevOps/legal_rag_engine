from __future__ import annotations

from typing import Protocol, Any


class LLMBackend(Protocol):
    def generate(self, query: str, context: str) -> str: ...


class ExampleLocalLLM:
    """Adapter interface only. Replace with Transformers/vLLM/Ollama/etc."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def generate(self, query: str, context: str) -> str:
        # Implement your selected LLM backend here.
        # Return only the final grounded answer string.
        raise NotImplementedError("Connect your LLM backend here.")
