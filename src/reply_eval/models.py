"""Typed result contracts shared by every evaluator and report renderer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


WEIGHTS = {
    "intent_accuracy": 0.25,
    "usefulness": 0.30,
    "groundedness": 0.30,
    "tone": 0.10,
    "clarity": 0.05,
}


@dataclass(frozen=True)
class MetricScore:
    score: float
    reason: str
    evidence: list[str]

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 100:
            raise ValueError("metric score must be between 0 and 100")
        if not self.reason.strip():
            raise ValueError("metric reason must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    metrics: dict[str, MetricScore]
    risk_tags: list[str]
    critical_fail: bool
    critical_reason: str | None
    improvement: str
    evaluator: dict[str, str]

    def __post_init__(self) -> None:
        if set(self.metrics) != set(WEIGHTS):
            raise ValueError(f"metrics must be exactly {sorted(WEIGHTS)}")

    @property
    def overall_score(self) -> float:
        weighted = sum(
            self.metrics[name].score * weight for name, weight in WEIGHTS.items()
        )
        if self.critical_fail:
            weighted = min(weighted, 59.0)
        return round(weighted, 2)

    def with_critical_failure(self, reason: str) -> "CaseEvaluation":
        return replace(self, critical_fail=True, critical_reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "metrics": {
                name: value.to_dict() for name, value in self.metrics.items()
            },
            "overall_score": self.overall_score,
            "risk_tags": list(self.risk_tags),
            "critical_fail": self.critical_fail,
            "critical_reason": self.critical_reason,
            "improvement": self.improvement,
            "evaluator": dict(self.evaluator),
        }


@dataclass(frozen=True)
class RunResult:
    metadata: dict[str, Any]
    summary: dict[str, Any]
    validation: dict[str, Any]
    cases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "validation": self.validation,
            "cases": self.cases,
        }
