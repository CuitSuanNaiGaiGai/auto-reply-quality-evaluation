import json
import tempfile
import unittest
from csv import DictReader
from pathlib import Path

from reply_eval.models import RunResult
from reply_eval.reporting import write_reports


class ReportingTests(unittest.TestCase):
    def test_rendered_text_files_have_no_trailing_whitespace(self):
        names = list(
            [
                "intent_accuracy",
                "usefulness",
                "groundedness",
                "tone",
                "clarity",
            ]
        )
        case = {
            "id": "case_01",
            "overall_score": 75,
            "user_question": "q",
            "auto_reply": "a",
            "metrics": {
                name: {"score": 75, "reason": "r", "evidence": ["e"]}
                for name in names
            },
            "risk_tags": [],
            "critical_fail": False,
            "critical_reason": None,
            "improvement": "improve",
            "evaluator": {"mode": "mock", "version": "1.0"},
        }
        result = RunResult(
            metadata={"judge_mode": "mock"},
            summary={},
            validation={},
            cases=[case],
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(result, Path(directory))
            for name, path in paths.items():
                with self.subTest(format=name):
                    lines = path.read_text(encoding="utf-8").splitlines()
                    self.assertFalse(
                        any(line != line.rstrip() for line in lines)
                    )

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

    def test_hybrid_reports_component_means_disagreements_and_call_counts(self):
        names = [
            "intent_accuracy",
            "usefulness",
            "groundedness",
            "tone",
            "clarity",
        ]
        cases = []
        for index, (mock_score, qwen_score, final_score) in enumerate(
            [(50, 60, 57), (50, 80, 71), (25, 75, 60)], start=1
        ):
            difference = abs(qwen_score - mock_score)
            cases.append(
                {
                    "id": f"case_{index:02d}",
                    "overall_score": final_score,
                    "mock_overall_score": mock_score,
                    "qwen_overall_score": qwen_score,
                    "user_question": "q",
                    "auto_reply": "a",
                    "metrics": {
                        name: {
                            "score": final_score,
                            "reason": "hybrid reason",
                            "evidence": ["hybrid evidence"],
                        }
                        for name in names
                    },
                    "mock_metrics": {
                        name: {
                            "score": mock_score,
                            "reason": "mock reason",
                            "evidence": ["mock evidence"],
                        }
                        for name in names
                    },
                    "qwen_metrics": {
                        name: {
                            "score": qwen_score,
                            "reason": "qwen reason",
                            "evidence": ["qwen evidence"],
                        }
                        for name in names
                    },
                    "judge_disagreement": {
                        **{name: difference for name in names},
                        "max": difference,
                    },
                    "qwen_request_count": 2 if index == 2 else 1,
                    "qwen_retry_count": 1 if index == 2 else 0,
                    "local_improvement_fallback": index == 3,
                    "risk_tags": [],
                    "critical_fail": False,
                    "critical_reason": None,
                    "improvement": "improve",
                    "evaluator": {"mode": "hybrid", "version": "1.0"},
                }
            )
        validation = {
            mode: {
                "sample_size": 3,
                "spearman_correlation": 0.5,
                "positive_negative_gap": 10.0,
                "issue_tag_match_rate": 0.5,
                "disagreements": [],
            }
            for mode in ("mock", "qwen", "hybrid")
        }
        result = RunResult(
            metadata={"judge_mode": "hybrid", "qwen_model": "qwen-test"},
            summary={},
            validation=validation,
            cases=cases,
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(result, Path(directory))
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            markdown = paths["markdown"].read_text(encoding="utf-8")
            html = paths["html"].read_text(encoding="utf-8")
            with paths["csv"].open(encoding="utf-8", newline="") as stream:
                csv_rows = list(DictReader(stream))

        summary = payload["summary"]
        self.assertEqual(summary["largest_disagreements"][0]["id"], "case_03")
        self.assertEqual(
            set(summary["component_overall_means"]),
            {"mock", "qwen", "hybrid"},
        )
        self.assertEqual(summary["request_count"], 4)
        self.assertEqual(summary["retry_count"], 1)
        self.assertEqual(summary["local_improvement_fallback_count"], 1)
        self.assertIn("Mock / Qwen / Hybrid", markdown)
        self.assertIn("最大分歧", html)
        self.assertIn("mock_overall_score", csv_rows[0])
        self.assertIn("qwen_overall_score", csv_rows[0])


if __name__ == "__main__":
    unittest.main()
