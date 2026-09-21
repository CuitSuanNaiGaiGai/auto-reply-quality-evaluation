import json
import tempfile
import unittest
from pathlib import Path

from reply_eval.models import RunResult
from reply_eval.reporting import write_reports


class ReportingTests(unittest.TestCase):
    def test_all_formats_share_summary_and_worst_three(self):
        names = [
            "intent_accuracy",
            "usefulness",
            "groundedness",
            "tone",
            "clarity",
        ]
        cases = [
            {
                "id": f"case_{index:02d}",
                "overall_score": score,
                "user_question": "q",
                "auto_reply": "a",
                "metrics": {
                    name: {"score": score, "reason": "r", "evidence": ["e"]}
                    for name in names
                },
                "risk_tags": [],
                "critical_fail": False,
                "critical_reason": None,
                "improvement": "improve",
                "evaluator": {"mode": "mock", "version": "1.0"},
            }
            for index, score in enumerate([30, 10, 20, 80], start=1)
        ]
        result = RunResult(
            metadata={"mode": "mock"},
            summary={"overall_mean": 35.0},
            validation={},
            cases=cases,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(result, Path(directory))
            self.assertEqual(set(paths), {"json", "csv", "markdown", "html"})
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(
                payload["summary"]["worst_three"],
                ["case_02", "case_03", "case_01"],
            )
            self.assertIn(
                "case_02", paths["markdown"].read_text(encoding="utf-8")
            )
            self.assertIn("case_02", paths["html"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
