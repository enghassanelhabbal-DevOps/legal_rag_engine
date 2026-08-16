from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.legal_ai.evaluation.baseline import PROTECTED_BASELINE, RetrievalBaseline
from src.legal_ai.evaluation.metrics import mean_mrr, mean_recall_at_k


@dataclass(frozen=True)
class GateResult:
    mrr: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float

    def as_baseline(self) -> RetrievalBaseline:
        return RetrievalBaseline(
            mrr=self.mrr,
            recall_at_1=self.recall_at_1,
            recall_at_3=self.recall_at_3,
            recall_at_5=self.recall_at_5,
            recall_at_10=self.recall_at_10,
            description="regression gate check",
        )


def _synthetic_results() -> list[dict[str, list[str]]]:
    return [
        {"retrieved": ["d1", "d2", "d3"], "relevant": ["d1"]},
        {"retrieved": ["d2", "d1", "d3"], "relevant": ["d2"]},
        {"retrieved": ["d3", "d2", "d1"], "relevant": ["d3"]},
    ]


def run_gate() -> GateResult:
    results = _synthetic_results()
    measured = GateResult(
        mrr=mean_mrr(results),
        recall_at_1=mean_recall_at_k(results, 1),
        recall_at_3=mean_recall_at_k(results, 3),
        recall_at_5=mean_recall_at_k(results, 5),
        recall_at_10=mean_recall_at_k(results, 10),
    )
    if not PROTECTED_BASELINE.check(measured.as_baseline()):
        raise RuntimeError(
            "Retrieval regression gate failed: measured metrics fell below protected baseline. "
            f"Measured={measured} | Baseline={PROTECTED_BASELINE}"
        )
    return measured


def main() -> None:
    result = run_gate()
    print(
        "Regression gate passed: "
        f"MRR={result.mrr:.3f}, "
        f"Recall@1={result.recall_at_1:.3f}, "
        f"Recall@3={result.recall_at_3:.3f}, "
        f"Recall@5={result.recall_at_5:.3f}, "
        f"Recall@10={result.recall_at_10:.3f}"
    )


if __name__ == "__main__":
    main()
