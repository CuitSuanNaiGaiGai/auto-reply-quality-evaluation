# Auto-Reply Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python pipeline that scores 20 customer-service auto-replies with an offline deterministic judge or optional Qwen judge and generates auditable reports and screenshots.

**Architecture:** A small `reply_eval` package separates input validation, score contracts, judge implementations, human-reference validation, and report rendering. Both judges return the same typed evaluation object; a CLI orchestrates a complete all-or-nothing run and writes JSON, CSV, Markdown, and self-contained HTML from one result model.

**Tech Stack:** Python 3.10+ standard library, `unittest`, OpenAI-compatible Qwen HTTP API via `urllib.request`, HTML/CSS/SVG for the visual report.

## Global Constraints

- Default execution must require no network, API key, or third-party Python package.
- Score the 20 records in `task3_auto_replies.json`; never use a case ID to choose a score.
- Human references are validation-only and must never enter a judge scoring request.
- Treat unsupported external claims as unverified, not proven false.
- Never store or echo `QWEN_API_KEY`.
- Qwen mode must fail explicitly instead of silently falling back to mock mode.
- Every case must contain five metric scores, evidence, risk tags, an overall score, and an improvement suggestion.
- Report formats must be generated from the same in-memory run result.
- Keep implementation and test dependencies in the Python standard library.

---

## Planned File Map

- `pyproject.toml`: package metadata and `reply-eval` command.
- `.gitignore`: secrets, Python caches, and generated transient files.
- `.env.example`: safe Qwen variable names without credentials.
- `src/reply_eval/models.py`: immutable score/result contracts and weighted-score rules.
- `src/reply_eval/io.py`: JSON loading and cross-file schema validation.
- `src/reply_eval/mock_judge.py`: deterministic rule-based scorer.
- `src/reply_eval/qwen_judge.py`: optional OpenAI-compatible Qwen client and strict response validation.
- `src/reply_eval/validation.py`: validation tiers and agreement statistics from human notes.
- `src/reply_eval/reporting.py`: JSON, CSV, Markdown, HTML, and SVG-backed chart rendering.
- `src/reply_eval/cli.py`: command-line orchestration and exit codes.
- `tests/`: focused unit and end-to-end tests.
- `outputs/`: generated evaluation artifacts.
- `screenshots/`: development and result screenshots.
- `README.md`: method, execution, findings, limitations, and AI-tool disclosure.

---

### Task 1: Package scaffold, result contracts, and input validation

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/reply_eval/__init__.py`
- Create: `src/reply_eval/models.py`
- Create: `src/reply_eval/io.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`
- Create: `tests/test_io.py`

**Interfaces:**
- Produces: `MetricScore`, `CaseEvaluation`, `RunResult`, `overall_score()`, `load_cases()`, and `load_human_references()`.
- `load_cases(path: Path) -> list[dict[str, str]]` validates `id`, `user_question`, and `auto_reply`.
- `load_human_references(path: Path, expected_ids: set[str]) -> list[dict[str, str]]` validates `human_reference`, `annotator_notes`, and exact ID alignment.

- [ ] **Step 1: Write failing model tests**

```python
# tests/test_models.py
import unittest
from reply_eval.models import MetricScore, CaseEvaluation


class ModelTests(unittest.TestCase):
    def test_weighted_score_and_critical_cap(self):
        metric = lambda score: MetricScore(score=score, reason="reason", evidence=["evidence"])
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_models -v`

Expected: import failure because `reply_eval.models` does not exist.

- [ ] **Step 3: Implement typed contracts and weighted scoring**

```python
# src/reply_eval/models.py
from dataclasses import dataclass, replace
from typing import Any

WEIGHTS = {
    "intent_accuracy": 0.25,
    "usefulness": 0.30,
    "groundedness": 0.30,
    "tone": 0.10,
    "clarity": 0.05,
}


@dataclass(frozen=True)
class MetricScore:
    score: float
    reason: str
    evidence: list[str]

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 100:
            raise ValueError("metric score must be between 0 and 100")
        if not self.reason.strip():
            raise ValueError("metric reason must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "reason": self.reason, "evidence": self.evidence}


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    metrics: dict[str, MetricScore]
    risk_tags: list[str]
    critical_fail: bool
    critical_reason: str | None
    improvement: str
    evaluator: dict[str, str]

    def __post_init__(self) -> None:
        if set(self.metrics) != set(WEIGHTS):
            raise ValueError(f"metrics must be exactly {sorted(WEIGHTS)}")

    @property
    def overall_score(self) -> float:
        weighted = sum(self.metrics[name].score * weight for name, weight in WEIGHTS.items())
        return round(min(weighted, 59.0) if self.critical_fail else weighted, 2)

    def with_critical_failure(self, reason: str) -> "CaseEvaluation":
        return replace(self, critical_fail=True, critical_reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "metrics": {name: value.to_dict() for name, value in self.metrics.items()},
            "overall_score": self.overall_score,
            "risk_tags": self.risk_tags,
            "critical_fail": self.critical_fail,
            "critical_reason": self.critical_reason,
            "improvement": self.improvement,
            "evaluator": self.evaluator,
        }


@dataclass(frozen=True)
class RunResult:
    metadata: dict[str, Any]
    summary: dict[str, Any]
    validation: dict[str, Any]
    cases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": self.summary,
            "validation": self.validation,
            "cases": self.cases,
        }
```

- [ ] **Step 4: Verify model tests pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_models -v`

Expected: 2 tests pass.

- [ ] **Step 5: Write failing input-validation tests**

```python
# tests/test_io.py
import json
import tempfile
import unittest
from pathlib import Path
from reply_eval.io import InputError, load_cases, load_human_references


class InputTests(unittest.TestCase):
    def write_json(self, payload):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "input.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_duplicate_case_ids_are_rejected(self):
        payload = [
            {"id": "case_01", "user_question": "q", "auto_reply": "a"},
            {"id": "case_01", "user_question": "q2", "auto_reply": "a2"},
        ]
        with self.assertRaisesRegex(InputError, "duplicate id: case_01"):
            load_cases(self.write_json(payload))

    def test_reference_ids_must_match_cases(self):
        payload = [{"id": "case_02", "human_reference": "r", "annotator_notes": "n"}]
        with self.assertRaisesRegex(InputError, "ID mismatch"):
            load_human_references(self.write_json(payload), {"case_01"})
```

- [ ] **Step 6: Run input tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_io -v`

Expected: import failure because `reply_eval.io` does not exist.

- [ ] **Step 7: Implement strict JSON loading**

Implement `InputError(ValueError)`, `_read_json_array()`, `_validate_string_fields()`, `load_cases()`, and `load_human_references()` in `src/reply_eval/io.py`. Error messages must include the file path and record index; duplicate IDs and cross-file ID mismatches must use the exact phrases asserted above.

- [ ] **Step 8: Add package metadata and safe environment template**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "reply-quality-eval"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[project.scripts]
reply-eval = "reply_eval.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

```dotenv
# .env.example
QWEN_API_KEY=
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
```

`.gitignore` must contain `.env`, `__pycache__/`, `*.pyc`, and `.DS_Store`, while keeping generated report artifacts and screenshots tracked as deliverables.

- [ ] **Step 9: Run all Task 1 tests**

Run: `PYTHONPATH=src python3 -m unittest tests.test_models tests.test_io -v`

Expected: 4 tests pass.

- [ ] **Step 10: Commit Task 1**

```bash
git add pyproject.toml .gitignore .env.example src/reply_eval tests/test_models.py tests/test_io.py tests/__init__.py
git commit -m "feat: add evaluation contracts and input validation"
```

---

### Task 2: Deterministic mock judge

**Files:**
- Create: `src/reply_eval/mock_judge.py`
- Create: `tests/test_mock_judge.py`

**Interfaces:**
- Consumes: `MetricScore` and `CaseEvaluation` from Task 1.
- Produces: `MockJudge.evaluate(case: dict[str, str], knowledge: str | None = None) -> CaseEvaluation`.
- Risk tags are drawn from `specific_case_unresolved`, `self_service_deflection`, `missing_clarification`, `multi_intent_missed`, `emotion_underaddressed`, `unsupported_claim`, and `unsupported_capability`.

- [ ] **Step 1: Write failing behavioral tests**

```python
# tests/test_mock_judge.py
import unittest
from reply_eval.mock_judge import MockJudge


class MockJudgeTests(unittest.TestCase):
    def setUp(self):
        self.judge = MockJudge()

    def evaluate(self, question, reply):
        return self.judge.evaluate({"id": "case_x", "user_question": question, "auto_reply": reply})

    def test_specific_case_deflection_lowers_usefulness(self):
        weak = self.evaluate("我的退款什么时候到账", "请在订单详情页查看，或联系客服。")
        strong = self.evaluate("我的退款什么时候到账", "请提供订单号，我帮您查询退款进度。")
        self.assertLess(weak.metrics["usefulness"].score, strong.metrics["usefulness"].score)
        self.assertIn("self_service_deflection", weak.risk_tags)

    def test_emotional_complaint_requires_empathy(self):
        cold = self.evaluate("等了20分钟都没人理我", "请说明您的问题。")
        empathic = self.evaluate("等了20分钟都没人理我", "非常抱歉让您久等了，请告诉我问题，我马上帮您处理。")
        self.assertLess(cold.metrics["tone"].score, empathic.metrics["tone"].score)

    def test_unsupported_numeric_policy_claim_is_flagged_not_declared_false(self):
        result = self.evaluate("能取消吗", "可以取消，退款会在1-3个工作日到账。")
        self.assertIn("unsupported_claim", result.risk_tags)
        self.assertIn("待核实", result.metrics["groundedness"].reason)

    def test_same_input_produces_identical_result(self):
        case = {"id": "case_x", "user_question": "优惠券怎么用不了", "auto_reply": "请提供优惠券编号，我帮您核实。"}
        self.assertEqual(self.judge.evaluate(case).to_dict(), self.judge.evaluate(case).to_dict())
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_mock_judge -v`

Expected: import failure because `reply_eval.mock_judge` does not exist.

- [ ] **Step 3: Implement generic signal extraction**

Create explicit regex/phrase tables for concrete-case markers, required-identifier scenarios, proactive service, self-service deflection, emotional scenarios, empathy, external claims, capability claims, and multiple intents. Implement helpers that return both boolean/count signals and the exact matched phrases so evidence is auditable.

The scorer must start every metric at a documented neutral baseline and apply bounded increments/decrements in 25-point units. It must not contain `case_01` through `case_20`, nor branch on the `id` field.

- [ ] **Step 4: Implement the scoring rubric**

Implement `MockJudge.evaluate()` with these fixed invariants:

```python
score = max(0, min(100, score))
```

- A direct answer or necessary clarification improves intent accuracy.
- A proactive handling offer or complete self-service path improves usefulness.
- Deflection in a concrete case reduces usefulness and adds a risk tag.
- Unsupported external claims reduce groundedness and are described as `待核实`, never `错误`.
- Emotion without matching empathy reduces tone.
- Long or internally focused text reduces clarity.
- High-risk fabricated completed actions set `critical_fail`; ordinary policy/time claims do not.

Every metric must include at least one evidence string, using `未检测到相关文本信号` when there is no literal match.

- [ ] **Step 5: Verify mock tests pass and scan for case-specific overfitting**

Run: `PYTHONPATH=src python3 -m unittest tests.test_mock_judge -v`

Run: `rg -n 'case_(0[1-9]|1[0-9]|20)' src/reply_eval/mock_judge.py`

Expected: 4 tests pass; `rg` returns no matches.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/reply_eval/mock_judge.py tests/test_mock_judge.py
git commit -m "feat: add deterministic mock quality judge"
```

---

### Task 3: Optional Qwen judge with strict contracts

**Files:**
- Create: `src/reply_eval/qwen_judge.py`
- Create: `tests/test_qwen_judge.py`

**Interfaces:**
- Consumes: `MetricScore` and `CaseEvaluation` from Task 1.
- Produces: `QwenConfig.from_env()`, `QwenJudge.evaluate(case, knowledge=None)`, `QwenError`.
- HTTP endpoint: `{QWEN_BASE_URL.rstrip('/')}/chat/completions`.

- [ ] **Step 1: Write failing Qwen configuration and response tests**

```python
# tests/test_qwen_judge.py
import json
import os
import unittest
from unittest.mock import patch
from reply_eval.qwen_judge import QwenConfig, QwenError, QwenJudge


VALID_CONTENT = json.dumps({
    "metrics": {
        name: {"score": 75, "reason": "reason", "evidence": ["evidence"]}
        for name in ["intent_accuracy", "usefulness", "groundedness", "tone", "clarity"]
    },
    "risk_tags": [],
    "critical_fail": False,
    "critical_reason": None,
    "improvement": "ask for the order id",
})


class QwenJudgeTests(unittest.TestCase):
    def test_missing_api_key_fails_without_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(QwenError, "QWEN_API_KEY"):
                QwenConfig.from_env()

    def test_valid_response_is_converted_to_case_evaluation(self):
        config = QwenConfig(api_key="secret", base_url="https://example.test/v1", model="qwen-test", timeout=10)
        transport = lambda request, timeout: {"choices": [{"message": {"content": VALID_CONTENT}}]}
        result = QwenJudge(config, transport=transport).evaluate(
            {"id": "case_x", "user_question": "q", "auto_reply": "a"}
        )
        self.assertEqual(result.overall_score, 75.0)
        self.assertEqual(result.evaluator["model"], "qwen-test")

    def test_invalid_json_is_reported_without_secret(self):
        config = QwenConfig(api_key="top-secret", base_url="https://example.test/v1", model="qwen-test", timeout=10)
        transport = lambda request, timeout: {"choices": [{"message": {"content": "not json"}}]}
        with self.assertRaises(QwenError) as caught:
            QwenJudge(config, transport=transport).evaluate({"id": "case_x", "user_question": "q", "auto_reply": "a"})
        self.assertNotIn("top-secret", str(caught.exception))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_qwen_judge -v`

Expected: import failure because `reply_eval.qwen_judge` does not exist.

- [ ] **Step 3: Implement configuration and HTTP transport**

Use frozen `QwenConfig` with `api_key`, `base_url`, `model`, and `timeout`. `from_env()` must read `QWEN_API_KEY`, optional `QWEN_BASE_URL`, and optional `QWEN_MODEL`; empty key raises `QwenError`. Build requests with `Authorization: Bearer ...` and JSON content type. Convert `HTTPError`, `URLError`, timeouts, malformed transport envelopes, and invalid model content to sanitized `QwenError` messages.

- [ ] **Step 4: Implement prompt version 1.0 and strict response parsing**

The system prompt must embed the five metric definitions and weights, state that human references are unavailable, prohibit treating model memory as platform evidence, require `unsupported_claim` for ungrounded external assertions, and request only JSON matching the result contract. Use `temperature: 0` and request a JSON object response format.

Validate exact metric names, 0–100 bounds, reason/evidence types, risk-tag list, boolean `critical_fail`, nullable critical reason, and non-empty improvement before constructing `CaseEvaluation`.

- [ ] **Step 5: Verify all Qwen tests pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_qwen_judge -v`

Expected: 3 tests pass without network access.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/reply_eval/qwen_judge.py tests/test_qwen_judge.py
git commit -m "feat: add optional qwen judge"
```

---

### Task 4: Human-reference validation and multi-format reporting

**Files:**
- Create: `src/reply_eval/validation.py`
- Create: `src/reply_eval/reporting.py`
- Create: `tests/test_validation.py`
- Create: `tests/test_reporting.py`

**Interfaces:**
- `build_validation(references, evaluations) -> dict[str, object]`.
- `build_summary(evaluations) -> dict[str, object]`.
- `write_reports(result: RunResult, output_dir: Path) -> dict[str, Path]`.

- [ ] **Step 1: Write failing validation-statistics tests**

```python
# tests/test_validation.py
import unittest
from reply_eval.validation import spearman_rank, summarize_tiers


class ValidationTests(unittest.TestCase):
    def test_spearman_is_one_for_matching_order(self):
        self.assertEqual(spearman_rank([1, 2, 3], [10, 20, 30]), 1.0)

    def test_positive_negative_gap_is_reported(self):
        rows = [
            {"tier": "positive", "score": 80},
            {"tier": "middle", "score": 60},
            {"tier": "negative", "score": 40},
        ]
        summary = summarize_tiers(rows)
        self.assertEqual(summary["positive_negative_gap"], 40.0)
```

- [ ] **Step 2: Run validation tests and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_validation -v`

Expected: import failure because `reply_eval.validation` does not exist.

- [ ] **Step 3: Implement transparent tiering and statistics**

Implement deterministic note classification using explicit positive and severe-negative phrases documented in the design. Store the matched phrase and derived tier for every reference. Compute average scores by tier, positive-negative mean gap, tie-aware Spearman correlation, expected issue-tag match rate, and a complete disagreement list. Return `None` with an explanation when a statistic is undefined instead of fabricating zero.

- [ ] **Step 4: Verify validation tests pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_validation -v`

Expected: 2 tests pass.

- [ ] **Step 5: Write failing reporting test**

```python
# tests/test_reporting.py
import json
import tempfile
import unittest
from pathlib import Path
from reply_eval.models import RunResult
from reply_eval.reporting import write_reports


class ReportingTests(unittest.TestCase):
    def test_all_formats_share_summary_and_worst_three(self):
        cases = [
            {"id": f"case_{index:02d}", "overall_score": score, "user_question": "q", "auto_reply": "a",
             "metrics": {name: {"score": score, "reason": "r", "evidence": ["e"]} for name in
                         ["intent_accuracy", "usefulness", "groundedness", "tone", "clarity"]},
             "risk_tags": [], "critical_fail": False, "critical_reason": None, "improvement": "improve",
             "evaluator": {"mode": "mock", "version": "1.0"}}
            for index, score in enumerate([30, 10, 20, 80], start=1)
        ]
        result = RunResult(metadata={"mode": "mock"}, summary={"overall_mean": 35.0}, validation={}, cases=cases)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(result, Path(directory))
            self.assertEqual(set(paths), {"json", "csv", "markdown", "html"})
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["worst_three"], ["case_02", "case_03", "case_01"])
            self.assertIn("case_02", paths["markdown"].read_text(encoding="utf-8"))
            self.assertIn("case_02", paths["html"].read_text(encoding="utf-8"))
```

- [ ] **Step 6: Run reporting test and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_reporting -v`

Expected: import failure because `reply_eval.reporting` does not exist.

- [ ] **Step 7: Implement summaries and report renderers**

Compute overall mean, quality band counts, metric mean/median/min/max, metric score bands, risk-tag counts, critical-fail count, and worst three sorted by `(overall_score, id)`. Write UTF-8 JSON, CSV, Markdown, and a self-contained HTML dashboard. The HTML must include semantic tables plus inline SVG bar charts; it must not load fonts, scripts, or styles from the network.

- [ ] **Step 8: Verify reporting tests pass**

Run: `PYTHONPATH=src python3 -m unittest tests.test_reporting -v`

Expected: 1 test passes and all four files exist.

- [ ] **Step 9: Commit Task 4**

```bash
git add src/reply_eval/validation.py src/reply_eval/reporting.py tests/test_validation.py tests/test_reporting.py
git commit -m "feat: add validation metrics and report renderers"
```

---

### Task 5: CLI, end-to-end run, README, and screenshots

**Files:**
- Create: `src/reply_eval/cli.py`
- Create: `tests/test_cli.py`
- Create: `README.md`
- Create: `outputs/evaluation_results.json`
- Create: `outputs/case_scores.csv`
- Create: `outputs/evaluation_report.md`
- Create: `outputs/evaluation_report.html`
- Create: `screenshots/development_process.png`
- Create: `screenshots/evaluation_results.png`

**Interfaces:**
- `main(argv: list[str] | None = None) -> int`.
- CLI options: `--input`, `--human-ref`, `--output-dir`, and `--judge {mock,qwen}`.
- Exit codes: `0` success, `2` input/configuration error, `3` judge/API error, `4` report-writing error.

- [ ] **Step 1: Write failing CLI end-to-end test**

```python
# tests/test_cli.py
import json
import tempfile
import unittest
from pathlib import Path
from reply_eval.cli import main


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_mock_run_scores_all_twenty_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            code = main([
                "--input", str(ROOT / "task3_auto_replies.json"),
                "--human-ref", str(ROOT / "task3_human_ref.json"),
                "--output-dir", directory,
                "--judge", "mock",
            ])
            self.assertEqual(code, 0)
            result = json.loads((Path(directory) / "evaluation_results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["cases"]), 20)
            self.assertEqual(len(result["summary"]["worst_three"]), 3)
            self.assertEqual(result["metadata"]["judge_mode"], "mock")
```

- [ ] **Step 2: Run CLI test and verify RED**

Run: `PYTHONPATH=src python3 -m unittest tests.test_cli -v`

Expected: import failure because `reply_eval.cli` does not exist.

- [ ] **Step 3: Implement all-or-nothing CLI orchestration**

Parse arguments, validate both inputs, construct the selected judge, evaluate all cases, enrich each result with the original question and reply, build the summary and validation block, then write all reports. Catch only expected input/configuration, judge, and report errors; print concise sanitized errors to standard error and return the documented exit code.

Do not load `task3_human_ref.json` into either judge. Pass it only to `build_validation()` after all case scores exist.

- [ ] **Step 4: Verify CLI test and full unit suite**

Run: `PYTHONPATH=src python3 -m unittest discover -s tests -v`

Expected: all tests pass with zero network requests.

- [ ] **Step 5: Generate the real mock evaluation artifacts**

Run:

```bash
PYTHONPATH=src python3 -m reply_eval.cli \
  --input task3_auto_replies.json \
  --human-ref task3_human_ref.json \
  --output-dir outputs \
  --judge mock
```

Expected: exit code 0; console states that 20 cases were scored and prints the four output paths.

- [ ] **Step 6: Inspect the generated conclusions for internal consistency**

Run:

```bash
python3 -m json.tool outputs/evaluation_results.json >/dev/null
python3 - <<'PY'
import csv, json
from pathlib import Path

payload = json.loads(Path("outputs/evaluation_results.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(Path("outputs/case_scores.csv").open(encoding="utf-8")))
assert len(payload["cases"]) == 20
assert len(rows) == 20
assert payload["summary"]["worst_three"] == [
    row["id"] for row in sorted(payload["cases"], key=lambda item: (item["overall_score"], item["id"]))[:3]
]
print("20 cases; JSON/CSV counts and worst-three ordering agree")
PY
```

Expected: `20 cases; JSON/CSV counts and worst-three ordering agree`.

- [ ] **Step 7: Write README from actual outputs**

README must contain:

- Project purpose and one-command mock quick start.
- Qwen setup commands using environment variables and no real key.
- Five metric definitions, weights, 0–4-to-100 conversion, and critical-fail rule.
- Clear distinction between intent accuracy and factual grounding.
- Explanation that human references validate but do not score cases.
- Actual overall score, metric summary, worst three, and validation findings read from generated output.
- Limitations and production improvements.
- AI-tool disclosure: Codex assisted with design, implementation, test generation, and documentation; deterministic outputs were verified by automated tests and local execution; no Qwen call was made unless the user supplied credentials and invoked Qwen mode.
- Links to both screenshots and report files.

- [ ] **Step 8: Capture genuine development and result screenshots**

Capture `screenshots/development_process.png` from the real terminal or agent development view showing test execution and project files. Capture `screenshots/evaluation_results.png` from the generated HTML report in a browser, showing the overall score, metric distribution, and worst-case section. Do not fabricate terminal output or edit score values into either image.

- [ ] **Step 9: Verify README commands and screenshot files**

Run the exact mock command copied from README into a fresh temporary output directory. Verify both PNG files using an image inspector and confirm their dimensions are large enough to read. Check README links resolve locally.

- [ ] **Step 10: Run final verification**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m reply_eval.cli --input task3_auto_replies.json --human-ref task3_human_ref.json --output-dir outputs --judge mock
git diff --check
git status --short
```

Expected: all tests pass; CLI exits 0 after scoring 20 cases; `git diff --check` reports no whitespace errors; status lists only intentional project deliverables.

- [ ] **Step 11: Commit final deliverables**

```bash
git add src/reply_eval/cli.py tests/test_cli.py README.md outputs screenshots pyproject.toml
git commit -m "feat: deliver auto-reply evaluation pipeline"
```

---

## Plan Self-Review Record

- Spec coverage: each design requirement maps to Tasks 1–5.
- Isolation: human references are only consumed by Task 4 validation after scoring.
- Type consistency: both judge implementations produce `CaseEvaluation`; reporting consumes `RunResult`.
- Reproducibility: mock mode is deterministic and dependency-free; Qwen mode records its configuration metadata.
- Safety: secrets are environment-only and errors are sanitized.
- Scope: no deployment or remote GitHub publication is included because the user did not request external publication.
