import json
import tempfile
import unittest
from pathlib import Path

from reply_eval.cli import main
from reply_eval.hybrid_judge import HybridJudge
from reply_eval.mock_judge import MockJudge
from reply_eval.models import CaseEvaluation, MetricScore, WEIGHTS


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_mock_run_scores_all_twenty_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            code = main(
                [
                    "--input",
                    str(ROOT / "task3_auto_replies.json"),
                    "--human-ref",
                    str(ROOT / "task3_human_ref.json"),
                    "--output-dir",
                    directory,
                    "--judge",
                    "mock",
                ]
            )
            self.assertEqual(code, 0)
            result = json.loads(
                (Path(directory) / "evaluation_results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(result["cases"]), 20)
            self.assertEqual(len(result["summary"]["worst_three"]), 3)
            self.assertEqual(result["metadata"]["judge_mode"], "mock")

    def test_missing_input_returns_input_error_code(self):
        with tempfile.TemporaryDirectory() as directory:
            code = main(
                [
                    "--input",
                    str(Path(directory) / "missing.json"),
                    "--output-dir",
                    directory,
                    "--judge",
                    "mock",
                ]
            )
            self.assertEqual(code, 2)

    def test_hybrid_run_keeps_component_scores_and_three_validations(self):
        class LocalQwenJudge:
            def evaluate(self, case, knowledge=None):
                return CaseEvaluation(
                    case_id=case["id"],
                    metrics={
                        name: MetricScore(
                            score=75,
                            reason="local semantic reason",
                            evidence=["local semantic evidence"],
                        )
                        for name in WEIGHTS
                    },
                    risk_tags=["qwen-local-risk"],
                    critical_fail=False,
                    critical_reason=None,
                    improvement="local semantic improvement",
                    evaluator={
                        "mode": "qwen",
                        "model": "qwen-local-test",
                        "version": "test-1",
                        "request_count": 1,
                        "retry_count": 0,
                        "local_improvement_fallback": False,
                    },
                )

        def factory(mode):
            self.assertEqual(mode, "hybrid")
            return HybridJudge(MockJudge(), LocalQwenJudge())

        with tempfile.TemporaryDirectory() as directory:
            code = main(
                [
                    "--input",
                    str(ROOT / "task3_auto_replies.json"),
                    "--human-ref",
                    str(ROOT / "task3_human_ref.json"),
                    "--output-dir",
                    directory,
                    "--judge",
                    "hybrid",
                ],
                judge_factory=factory,
            )

            self.assertEqual(code, 0)
            result = json.loads(
                (Path(directory) / "evaluation_results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["metadata"]["judge_mode"], "hybrid")
            self.assertEqual(result["metadata"]["qwen_model"], "qwen-local-test")
            self.assertEqual(result["metadata"]["request_count"], 20)
            self.assertEqual(len(result["cases"]), 20)
            self.assertIn("mock_metrics", result["cases"][0])
            self.assertIn("qwen_metrics", result["cases"][0])
            self.assertEqual(
                set(result["validation"]), {"mock", "qwen", "hybrid"}
            )


if __name__ == "__main__":
    unittest.main()
