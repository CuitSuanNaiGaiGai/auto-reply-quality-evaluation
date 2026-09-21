import unittest

from reply_eval.hybrid_judge import HybridJudge
from reply_eval.models import CaseEvaluation, MetricScore, WEIGHTS


CASE = {"id": "case_x", "user_question": "q", "auto_reply": "a"}


def evaluation(
    score: float,
    *,
    mode: str,
    tags: list[str] | None = None,
    critical: bool = False,
    improvement: str | None = None,
    fallback: bool = False,
) -> CaseEvaluation:
    return CaseEvaluation(
        case_id="case_x",
        metrics={
            name: MetricScore(
                score=score,
                reason=f"{mode} reason",
                evidence=[f"{mode} evidence"],
            )
            for name in WEIGHTS
        },
        risk_tags=tags or [],
        critical_fail=critical,
        critical_reason=f"{mode} critical" if critical else None,
        improvement=improvement or f"{mode} improvement",
        evaluator={
            "mode": mode,
            "version": f"{mode}-1",
            "model": "qwen-test" if mode == "qwen" else None,
            "request_count": 2 if mode == "qwen" else 0,
            "retry_count": 1 if mode == "qwen" else 0,
            "local_improvement_fallback": fallback,
        },
    )


class StaticJudge:
    def __init__(self, result: CaseEvaluation):
        self.result = result

    def evaluate(self, case, knowledge=None):
        return self.result


class HybridJudgeTests(unittest.TestCase):
    def test_fuses_each_metric_seventy_thirty(self):
        mock = evaluation(50, mode="mock", tags=["shared", "mock-risk"])
        qwen = evaluation(100, mode="qwen", tags=["shared", "qwen-risk"])

        result = HybridJudge(StaticJudge(mock), StaticJudge(qwen)).evaluate(CASE)

        self.assertEqual(result.final.metrics["usefulness"].score, 85.0)
        self.assertEqual(
            result.final.risk_tags,
            ["shared", "mock-risk", "qwen-risk"],
        )
        self.assertEqual(result.max_disagreement, 50.0)
        self.assertIn("Qwen：qwen reason", result.final.metrics["tone"].reason)
        self.assertIn("Mock：mock reason", result.final.metrics["tone"].reason)

    def test_critical_failure_from_either_judge_is_preserved(self):
        result = HybridJudge(
            StaticJudge(evaluation(100, mode="mock", critical=True)),
            StaticJudge(evaluation(100, mode="qwen")),
        ).evaluate(CASE)

        self.assertTrue(result.final.critical_fail)
        self.assertIn("mock critical", result.final.critical_reason)
        self.assertEqual(result.final.overall_score, 59.0)

    def test_mock_improvement_is_used_when_qwen_used_local_fallback(self):
        result = HybridJudge(
            StaticJudge(
                evaluation(50, mode="mock", improvement="mock suggestion")
            ),
            StaticJudge(
                evaluation(
                    75,
                    mode="qwen",
                    improvement="local template",
                    fallback=True,
                )
            ),
        ).evaluate(CASE)

        self.assertEqual(result.final.improvement, "mock suggestion")

    def test_serialization_retains_components_and_request_metadata(self):
        result = HybridJudge(
            StaticJudge(evaluation(25, mode="mock")),
            StaticJudge(evaluation(75, mode="qwen")),
        ).evaluate(CASE)

        rendered = result.to_case_dict()

        self.assertEqual(rendered["mock_metrics"]["tone"]["score"], 25)
        self.assertEqual(rendered["qwen_metrics"]["tone"]["score"], 75)
        self.assertEqual(rendered["metrics"]["tone"]["score"], 60.0)
        self.assertEqual(rendered["mock_overall_score"], 25.0)
        self.assertEqual(rendered["qwen_overall_score"], 75.0)
        self.assertEqual(rendered["judge_disagreement"]["tone"], 50.0)
        self.assertEqual(rendered["judge_disagreement"]["max"], 50.0)
        self.assertEqual(rendered["qwen_request_count"], 2)
        self.assertEqual(rendered["qwen_retry_count"], 1)
        self.assertFalse(rendered["local_improvement_fallback"])


if __name__ == "__main__":
    unittest.main()
