"""Optional Qwen judge through Alibaba Cloud's OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import CaseEvaluation, MetricScore, WEIGHTS


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
PROMPT_VERSION = "qwen-rubric-1.0"


class QwenError(RuntimeError):
    """Raised for configuration, transport, or response-contract failures."""


@dataclass(frozen=True)
class QwenConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "QwenConfig":
        api_key = os.environ.get("QWEN_API_KEY", "").strip()
        if not api_key:
            raise QwenError(
                "QWEN_API_KEY is required for --judge qwen; mock mode was not used"
            )
        base_url = os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL).strip()
        model = os.environ.get("QWEN_MODEL", DEFAULT_MODEL).strip()
        if not base_url or not model:
            raise QwenError("QWEN_BASE_URL and QWEN_MODEL must not be empty")
        return cls(api_key=api_key, base_url=base_url, model=model)


SYSTEM_PROMPT = """
你是客服自动回复质量评估器。仅使用用户问题、自动回复、本量表与可选知识上下文。
人工参考答案不可用。不要用模型记忆宣称平台政策、商品信息或已执行操作属实。
对缺少当前证据的外部事实标记 unsupported_claim，说明“待核实”，不要断言其错误。

对五项指标分别给出 0/25/50/75/100 之一：
1. intent_accuracy（25%）：核心诉求、多意图、必要追问。
2. usefulness（30%）：可执行下一步、服务闭环、不推责。
3. groundedness（30%）：政策、时效、金额、参数和能力声明的证据。
4. tone（10%）：礼貌、共情、情绪强度匹配。
5. clarity（5%）：简洁、结构、无无关内部话术。

只返回 JSON 对象，不要 Markdown。格式：
{
  "metrics": {
    "intent_accuracy": {"score": 0, "reason": "", "evidence": [""]},
    "usefulness": {"score": 0, "reason": "", "evidence": [""]},
    "groundedness": {"score": 0, "reason": "", "evidence": [""]},
    "tone": {"score": 0, "reason": "", "evidence": [""]},
    "clarity": {"score": 0, "reason": "", "evidence": [""]}
  },
  "risk_tags": [],
  "critical_fail": false,
  "critical_reason": null,
  "improvement": ""
}
只有高风险虚构承诺、错误安全建议或宣称已执行无证据操作时才将 critical_fail 设为 true。
""".strip()


Transport = Callable[[Request, int], dict[str, Any]]


def _default_transport(request: Request, timeout: int) -> dict[str, Any]:
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("top-level HTTP response is not an object")
    return payload


class QwenJudge:
    def __init__(self, config: QwenConfig, transport: Transport | None = None):
        self.config = config
        self.transport = transport or _default_transport

    def evaluate(
        self, case: dict[str, str], knowledge: str | None = None
    ) -> CaseEvaluation:
        user_payload: dict[str, str] = {
            "user_question": case["user_question"],
            "auto_reply": case["auto_reply"],
        }
        if knowledge:
            user_payload["knowledge_context"] = knowledge
        request_payload = {
            "model": self.config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
        }
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        request = Request(
            endpoint,
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            response = self.transport(request, self.config.timeout)
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content is not text")
            parsed = json.loads(content)
            return self._parse(case["id"], parsed)
        except QwenError:
            raise
        except HTTPError as exc:
            self._raise_sanitized(f"Qwen HTTP error {exc.code}")
        except (URLError, socket.timeout, TimeoutError) as exc:
            self._raise_sanitized(f"Qwen network error: {exc}")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._raise_sanitized(f"Qwen returned an invalid response: {exc}")
        raise AssertionError("unreachable")

    def _raise_sanitized(self, message: str) -> None:
        safe = message.replace(self.config.api_key, "[REDACTED]")
        raise QwenError(safe)

    def _parse(self, case_id: str, payload: Any) -> CaseEvaluation:
        if not isinstance(payload, dict):
            raise ValueError("judge result must be an object")
        raw_metrics = payload.get("metrics")
        if not isinstance(raw_metrics, dict) or set(raw_metrics) != set(WEIGHTS):
            raise ValueError(f"metrics must be exactly {sorted(WEIGHTS)}")
        metrics: dict[str, MetricScore] = {}
        for name in WEIGHTS:
            item = raw_metrics[name]
            if not isinstance(item, dict):
                raise ValueError(f"metric {name} must be an object")
            score = item.get("score")
            reason = item.get("reason")
            evidence = item.get("evidence")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(f"metric {name} score must be numeric")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"metric {name} reason must be non-empty text")
            if not isinstance(evidence, list) or not all(
                isinstance(value, str) for value in evidence
            ):
                raise ValueError(f"metric {name} evidence must be a text array")
            metrics[name] = MetricScore(float(score), reason, evidence)

        risk_tags = payload.get("risk_tags")
        critical_fail = payload.get("critical_fail")
        critical_reason = payload.get("critical_reason")
        improvement = payload.get("improvement")
        if not isinstance(risk_tags, list) or not all(
            isinstance(value, str) for value in risk_tags
        ):
            raise ValueError("risk_tags must be a text array")
        if not isinstance(critical_fail, bool):
            raise ValueError("critical_fail must be boolean")
        if critical_reason is not None and not isinstance(critical_reason, str):
            raise ValueError("critical_reason must be text or null")
        if critical_fail and not critical_reason:
            raise ValueError("critical_reason is required when critical_fail is true")
        if not isinstance(improvement, str) or not improvement.strip():
            raise ValueError("improvement must be non-empty text")
        return CaseEvaluation(
            case_id=case_id,
            metrics=metrics,
            risk_tags=risk_tags,
            critical_fail=critical_fail,
            critical_reason=critical_reason,
            improvement=improvement,
            evaluator={
                "mode": "qwen",
                "model": self.config.model,
                "version": PROMPT_VERSION,
            },
        )
