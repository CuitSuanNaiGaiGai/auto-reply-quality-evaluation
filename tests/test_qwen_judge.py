import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

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
    def setUp(self):
        self.config = QwenConfig(
            api_key="test-secret-value",
            base_url="https://example.test/v1",
            model="qwen-test",
            timeout=10,
        )
        self.case = {
            "id": "case_x",
            "user_question": "q",
            "auto_reply": "a",
        }

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

    def test_empty_improvement_uses_local_fallback_without_retry(self):
        payload = json.loads(VALID_CONTENT)
        payload["improvement"] = ""
        calls = []

        def transport(request, timeout):
            calls.append(request)
            return {
                "choices": [
                    {"message": {"content": json.dumps(payload)}}
                ]
            }

        result = QwenJudge(self.config, transport=transport).evaluate(self.case)

        self.assertEqual(len(calls), 1)
        self.assertTrue(result.improvement)
        self.assertTrue(result.evaluator["local_improvement_fallback"])
        self.assertEqual(result.evaluator["request_count"], 1)
        self.assertEqual(result.evaluator["retry_count"], 0)

    def test_invalid_json_retries_once_then_succeeds(self):
        responses = iter(
            [
                {"choices": [{"message": {"content": "not json"}}]},
                {"choices": [{"message": {"content": VALID_CONTENT}}]},
            ]
        )
        requests = []

        def transport(request, timeout):
            requests.append(json.loads(request.data))
            return next(responses)

        result = QwenJudge(self.config, transport=transport).evaluate(self.case)

        self.assertEqual(result.evaluator["request_count"], 2)
        self.assertEqual(result.evaluator["retry_count"], 1)
        repair_message = requests[1]["messages"][-1]["content"]
        self.assertIn("修复", repair_message)
        self.assertNotIn(self.config.api_key, repair_message)

    def test_empty_evidence_retries_as_contract_error(self):
        payload = json.loads(VALID_CONTENT)
        payload["metrics"]["tone"]["evidence"] = [""]
        responses = iter(
            [
                {
                    "choices": [
                        {"message": {"content": json.dumps(payload)}}
                    ]
                },
                {"choices": [{"message": {"content": VALID_CONTENT}}]},
            ]
        )

        result = QwenJudge(
            self.config, transport=lambda request, timeout: next(responses)
        ).evaluate(self.case)

        self.assertEqual(result.evaluator["request_count"], 2)

    def test_http_404_does_not_retry_and_names_model_without_secret(self):
        calls = 0

        def transport(request, timeout):
            nonlocal calls
            calls += 1
            raise HTTPError(request.full_url, 404, "missing", {}, None)

        with self.assertRaises(QwenError) as caught:
            QwenJudge(self.config, transport=transport).evaluate(self.case)

        message = str(caught.exception)
        self.assertEqual(calls, 1)
        self.assertIn("QWEN_MODEL", message)
        self.assertIn("qwen-test", message)
        self.assertIn("case_x", message)
        self.assertNotIn(self.config.api_key, message)

    def test_http_auth_errors_do_not_retry(self):
        for status in (401, 403):
            with self.subTest(status=status):
                calls = []

                def transport(request, timeout):
                    calls.append(request)
                    raise HTTPError(
                        request.full_url, status, "unauthorized", {}, None
                    )

                with self.assertRaisesRegex(QwenError, "密钥或模型权限"):
                    QwenJudge(self.config, transport=transport).evaluate(
                        self.case
                    )
                self.assertEqual(len(calls), 1)

    def test_http_429_retries_once_then_succeeds(self):
        calls = 0

        def transport(request, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise HTTPError(request.full_url, 429, "busy", {}, None)
            return {"choices": [{"message": {"content": VALID_CONTENT}}]}

        result = QwenJudge(self.config, transport=transport).evaluate(self.case)

        self.assertEqual(calls, 2)
        self.assertEqual(result.evaluator["retry_count"], 1)

    def test_http_5xx_stops_after_one_retry(self):
        calls = 0

        def transport(request, timeout):
            nonlocal calls
            calls += 1
            raise HTTPError(request.full_url, 503, "busy", {}, None)

        with self.assertRaises(QwenError) as caught:
            QwenJudge(self.config, transport=transport).evaluate(self.case)

        message = str(caught.exception)
        self.assertEqual(calls, 2)
        self.assertIn("case_x", message)
        self.assertIn("attempts=2", message)
        self.assertNotIn(self.config.api_key, message)


if __name__ == "__main__":
    unittest.main()
