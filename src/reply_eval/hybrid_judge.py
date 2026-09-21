"""Deterministic fusion of the semantic Qwen judge and stable Mock judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import CaseEvaluation, MetricScore, WEIGHTS


QWEN_WEIGHT = 0.70
MOCK_WEIGHT = 0.30
FUSION_VERSION = "hybrid-fusion-1.0"


class Judge(Protocol):
    def evaluate(
        self, case: dict[str, str], knowledge: str | None = None
    ) -> CaseEvaluation: ...


@dataclass(frozen=True)
class HybridEvaluation:
    final: CaseEvaluation
    mock: CaseEvaluation
    qwen: CaseEvaluation
    disagreement: dict[str, float]

    @property
    def max_disagreement(self) -> float:
        return max(self.disagreement.values(), default=0.0)

    def to_case_dict(self) -> dict[str, Any]:
        payload = self.final.to_dict()
        payload.update(
            {
                "mock_metrics": {
                    name: value.to_dict()
                    for name, value in self.mock.metrics.items()
                },
                "qwen_metrics": {
                    name: value.to_dict()
                    for name, value in self.qwen.metrics.items()
                },
                "mock_overall_score": self.mock.overall_score,
                "qwen_overall_score": self.qwen.overall_score,
                "judge_disagreement": {
                    **self.disagreement,
                    "max": self.max_disagreement,
                },
                "qwen_request_count": int(
                    self.qwen.evaluator.get("request_count", 0)
                ),
                "qwen_retry_count": int(
                    self.qwen.evaluator.get("retry_count", 0)
                ),
                "local_improvement_fallback": bool(
                    self.qwen.evaluator.get(
                        "local_improvement_fallback", False
                    )
                ),
            }
        )
        return payload


class HybridJudge:
    """Run both judges and combine their results without hiding either one."""

    def __init__(self, mock_judge: Judge, qwen_judge: Judge):
        self.mock_judge = mock_judge
        self.qwen_judge = qwen_judge

    def evaluate(
        self, case: dict[str, str], knowledge: str | None = None
    ) -> HybridEvaluation:
        mock = self.mock_judge.evaluate(case, knowledge)
        qwen = self.qwen_judge.evaluate(case, knowledge)
        metrics: dict[str, MetricScore] = {}
        disagreement: dict[str, float] = {}
        for name in WEIGHTS:
            mock_metric = mock.metrics[name]
            qwen_metric = qwen.metrics[name]
            metrics[name] = MetricScore(
                score=round(
                    qwen_metric.score * QWEN_WEIGHT
                    + mock_metric.score * MOCK_WEIGHT,
                    2,
                ),
                reason=(
                    f"Qwen：{qwen_metric.reason}；"
                    f"Mock：{mock_metric.reason}"
                ),
                evidence=[
                    *(f"Qwen: {item}" for item in qwen_metric.evidence),
                    *(f"Mock: {item}" for item in mock_metric.evidence),
                ],
            )
            disagreement[name] = round(
                abs(qwen_metric.score - mock_metric.score), 2
            )

        critical_reasons = [
            value.critical_reason
            for value in (mock, qwen)
            if value.critical_fail and value.critical_reason
        ]
        qwen_used_fallback = bool(
            qwen.evaluator.get("local_improvement_fallback", False)
        )
        improvement = mock.improvement if qwen_used_fallback else qwen.improvement
        final = CaseEvaluation(
            case_id=case["id"],
            metrics=metrics,
            risk_tags=list(dict.fromkeys([*mock.risk_tags, *qwen.risk_tags])),
            critical_fail=mock.critical_fail or qwen.critical_fail,
            critical_reason="；".join(critical_reasons) or None,
            improvement=improvement,
            evaluator={
                "mode": "hybrid",
                "version": FUSION_VERSION,
                "mock_version": mock.evaluator.get("version"),
                "qwen_version": qwen.evaluator.get("version"),
                "qwen_model": qwen.evaluator.get("model"),
            },
        )
        return HybridEvaluation(
            final=final,
            mock=mock,
            qwen=qwen,
            disagreement=disagreement,
        )
