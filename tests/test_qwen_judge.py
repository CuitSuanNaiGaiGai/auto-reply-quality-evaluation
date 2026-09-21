import json
import os
import unittest
from unittest.mock import patch

from reply_eval.qwen_judge import QwenConfig, QwenError, QwenJudge


VALID_CONTENT = json.dumps(
    {
        "metrics": {
            name: {"score": 75, "reason": "reason", "evidence": ["evidence"]}
            for name in [
                "intent_accuracy",
                "usefulness",
                "groundedness",
                "tone",
                "clarity",
            ]
        },
        "risk_tags": [],
        "critical_fail": False,
        "critical_reason": None,
        "improvement": "ask for the order id",
    }
)


class QwenJudgeTests(unittest.TestCase):
    def test_missing_api_key_fails_without_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(QwenError, "QWEN_API_KEY"):
                QwenConfig.from_env()

    def test_valid_response_is_converted_to_case_evaluation(self):
        config = QwenConfig(
            api_key="secret",
            base_url="https://example.test/v1",
            model="qwen-test",
            timeout=10,
        )
        transport = lambda request, timeout: {
            "choices": [{"message": {"content": VALID_CONTENT}}]
        }
        result = QwenJudge(config, transport=transport).evaluate(
            {"id": "case_x", "user_question": "q", "auto_reply": "a"}
        )
        self.assertEqual(result.overall_score, 75.0)
        self.assertEqual(result.evaluator["model"], "qwen-test")

    def test_invalid_json_is_reported_without_secret(self):
        config = QwenConfig(
            api_key="top-secret",
            base_url="https://example.test/v1",
            model="qwen-test",
            timeout=10,
        )
        transport = lambda request, timeout: {
            "choices": [{"message": {"content": "not json"}}]
        }
        with self.assertRaises(QwenError) as caught:
            QwenJudge(config, transport=transport).evaluate(
                {"id": "case_x", "user_question": "q", "auto_reply": "a"}
            )
        self.assertNotIn("top-secret", str(caught.exception))

    def test_reference_text_never_enters_request(self):
        config = QwenConfig(
            api_key="secret",
            base_url="https://example.test/v1",
            model="qwen-test",
            timeout=10,
        )
        captured = {}

        def transport(request, timeout):
            captured["body"] = json.loads(request.data)
            return {"choices": [{"message": {"content": VALID_CONTENT}}]}

        QwenJudge(config, transport=transport).evaluate(
            {"id": "case_x", "user_question": "question", "auto_reply": "reply"}
        )
        body = json.dumps(captured["body"], ensure_ascii=False)
        self.assertNotIn("human_reference", body)
        self.assertNotIn("annotator_notes", body)


if __name__ == "__main__":
    unittest.main()
