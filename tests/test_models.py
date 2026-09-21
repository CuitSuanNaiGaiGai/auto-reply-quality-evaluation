import unittest

from reply_eval.models import CaseEvaluation, MetricScore


class ModelTests(unittest.TestCase):
    def test_weighted_score_and_critical_cap(self):
        def metric(score: float) -> MetricScore:
            return MetricScore(score=score, reason="reason", evidence=["evidence"])

        normal = CaseEvaluation(
            case_id="case_01",
            metrics={
                "intent_accuracy": metric(100),
                "usefulness": metric(50),
                "groundedness": metric(100),
                "tone": metric(100),
                "clarity": metric(100),
            },
            risk_tags=[],
            critical_fail=False,
            critical_reason=None,
            improvement="ask for the order id",
            evaluator={"mode": "mock", "version": "1.0"},
        )
        self.assertEqual(normal.overall_score, 85.0)

        critical = normal.with_critical_failure("unsafe fabricated action")
        self.assertEqual(critical.overall_score, 59.0)

    def test_metric_rejects_out_of_range_score(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            MetricScore(score=101, reason="reason", evidence=[])


if __name__ == "__main__":
    unittest.main()
