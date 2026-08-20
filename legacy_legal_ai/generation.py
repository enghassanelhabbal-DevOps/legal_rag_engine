from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("legal_ai.generation")


class LLMManager:
    """Adapter that selects an available LLM backend implementation.

    Preference order: qwen_transformers.QwenTransformersBackend -> llm_backend_template.ExampleLocalLLM
    Implements basic prompt-injection protection and prompt scaffolding.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.backend = None
        self.actual_backend = None

    def _detect_prompt_injection(self, text: str) -> bool:
        # Very conservative heuristic: look for instruction-like tokens that
        # attempt to override system instructions. This is intentionally simple
        # and should be improved with a proper sanitizer later.
        suspicious = ["you are", "ignore the above", "system:", "assistant:", "do not" ]
        t = text.lower()
        return any(s in t for s in suspicious)

    def _build_system_prompt(self) -> str:
        return (
            "أنت نظام للبحث القانوني. استخدم فقط النصوص القانونية المقدمة في 'السياق'. "
            "لا تختَرع مواداً أو أرقام مواد أو أحكامًا. لكل استنتاج، قدم إشارة مصدرية. "
            "إذا كان الدليل غير كافٍ، أبلغ بوضوح 'insufficient evidence'."
        )

    def load(self) -> None:
        try:
            from qwen_transformers_backend import QwenConfig, QwenTransformersBackend

            qconf = QwenConfig()
            if self.config:
                # shallow apply
                for k, v in self.config.items():
                    if hasattr(qconf, k):
                        setattr(qconf, k, v)
            self.backend = QwenTransformersBackend(qconf)
            self.backend.load()
            self.actual_backend = "qwen_transformers"
            LOGGER.info("Using QwenTransformersBackend")
            return
        except Exception as exc:
            LOGGER.warning("Qwen backend unavailable: %s", exc)

        try:
            from llm_backend_template import ExampleLocalLLM

            # Fallback: ExampleLocalLLM requires a model instance; pass None and expect user to patch.
            self.backend = ExampleLocalLLM(model=None)
            self.actual_backend = "example_local"
            LOGGER.info("Using ExampleLocalLLM fallback")
            return
        except Exception as exc:
            LOGGER.error("No available LLM backend: %s", exc)
            raise RuntimeError("No LLM backend available")

    def unload(self) -> None:
        if self.backend is None:
            return
        try:
            if hasattr(self.backend, "unload"):
                self.backend.unload()
        except Exception:
            pass
        self.backend = None

    def generate(self, query: str, context: str) -> str:
        # Basic injection detection
        if self._detect_prompt_injection(query) or self._detect_prompt_injection(context):
            raise RuntimeError("Prompt rejected: potential prompt injection detected.")

        system = self._build_system_prompt()
        # Build a compact structured prompt that instructs the LLM to cite sources
        prompt = (
            f"SYSTEM:\n{system}\n\n"
            f"LEGAL_CONTEXT:\n{context}\n\n"
            f"USER_QUERY:\n{query}\n\n"
            "INSTRUCTIONS:\nAnswer using ONLY the LEGAL_CONTEXT. Cite sources for every legal claim. "
            "If evidence is insufficient, say 'insufficient evidence'. Return JSON with keys: answer, citations, evidence, confidence, warnings."
        )

        if self.backend is None:
            self.load()

        # Delegate to backend. Backends are expected to accept (query, context) or raw prompt depending on implementation.
        # For backward compatibility, prefer generate(query, context) if available; otherwise pass prompt in place of context.
        if hasattr(self.backend, "generate"):
            try:
                return self.backend.generate(query, context)
            except TypeError:
                # Some backends may expect a single prompt string
                return self.backend.generate(prompt, "")
        raise RuntimeError("Loaded backend does not implement generate()")

    def info(self) -> dict[str, Any]:
        if self.backend is None:
            return {"backend": None}
        if hasattr(self.backend, "info"):
            try:
                return self.backend.info()
            except Exception:
                return {"backend": self.actual_backend}
        return {"backend": self.actual_backend}


__all__ = ["LLMManager"]
