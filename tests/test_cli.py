import json
import tempfile
import unittest
from pathlib import Path

from reply_eval.cli import main


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


if __name__ == "__main__":
    unittest.main()
