"""manager.py — LLM adapter manager.

Migrated from legal_ai/generation.py.
Selects the best available LLM backend (Qwen → fallback template).
Implements prompt scaffolding and basic injection detection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.legal_ai.core.logging import get_logger
from src.legal_ai.core.exceptions import GenerationError

LOGGER = get_logger(__name__)

# Tokens that suggest a prompt-injection attempt (conservative heuristic)
_INJECTION_PATTERNS = [
    "you are",
    "ignore the above",
    "system:",
    "assistant:",
    "do not",
]

_SYSTEM_PROMPT = (
    "أنت نظام للبحث القانوني. استخدم فقط النصوص القانونية المقدمة في 'السياق'. "
    "لا تختَرع مواداً أو أرقام مواد أو أحكامًا. لكل استنتاج، قدم إشارة مصدرية. "
    "إذا كان الدليل غير كافٍ، أبلغ بوضوح 'insufficient evidence'."
)


class LLMManager:
    """Selects and wraps an available LLM backend implementation.

    Preference order:
      1. qwen_transformers_backend.QwenTransformersBackend
      2. llm_backend_template.ExampleLocalLLM  (fallback stub)

    Implements:
      - Prompt-injection detection (conservative heuristic)
      - Structured prompt scaffolding with system instruction
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.backend: Any = None
        self.actual_backend: Optional[str] = None

    # ------------------------------------------------------------------
    # Security helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_prompt_injection(text: str) -> bool:
        t = text.lower()
        return any(p in t for p in _INJECTION_PATTERNS)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return (
            f"SYSTEM:\n{_SYSTEM_PROMPT}\n\n"
            f"LEGAL_CONTEXT:\n{context}\n\n"
            f"USER_QUERY:\n{query}\n\n"
            "INSTRUCTIONS:\nAnswer using ONLY the LEGAL_CONTEXT. "
            "Cite sources for every legal claim. "
            "If evidence is insufficient, say 'insufficient evidence'. "
            "Return JSON with keys: answer, citations, evidence, confidence, warnings."
        )

    # ------------------------------------------------------------------
    # Backend lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the best available backend.  Raises GenerationError if none found."""
        try:
            from qwen_transformers_backend import QwenConfig, QwenTransformersBackend  # type: ignore

            qconf = QwenConfig()
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
            from llm_backend_template import ExampleLocalLLM  # type: ignore

            self.backend = ExampleLocalLLM(model=None)
            self.actual_backend = "example_local"
            LOGGER.info("Using ExampleLocalLLM fallback")
            return
        except Exception as exc:
            LOGGER.error("No available LLM backend: %s", exc)
            raise GenerationError("No LLM backend available") from exc

    def unload(self) -> None:
        if self.backend is None:
            return
        try:
            if hasattr(self.backend, "unload"):
                self.backend.unload()
        except Exception:
            pass
        finally:
            self.backend = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, query: str, context: str) -> str:
        if self._detect_prompt_injection(query) or self._detect_prompt_injection(context):
            raise GenerationError("Prompt rejected: potential prompt injection detected.")

        if self.backend is None:
            self.load()

        prompt = self._build_prompt(query, context)

        if hasattr(self.backend, "generate"):
            try:
                return self.backend.generate(query, context)
            except TypeError:
                return self.backend.generate(prompt, "")

        raise GenerationError("Loaded backend does not implement generate()")

    def info(self) -> Dict[str, Any]:
        if self.backend is None:
            return {"backend": None}
        if hasattr(self.backend, "info"):
            try:
                return self.backend.info()
            except Exception:
                pass
        return {"backend": self.actual_backend}


__all__ = ["LLMManager"]
