# Hybrid Qwen Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a robust `hybrid` judge that fuses Qwen and deterministic Mock scores, tolerates non-critical Qwen formatting defects, loads Qwen settings safely from `.env`, and exposes auditable component scores and disagreements in every report.

**Architecture:** Keep `MockJudge` and `QwenJudge` independent, introduce a small `HybridJudge` composition layer, and keep orchestration in `cli.py`. Qwen response recovery and request accounting remain inside `QwenJudge`; report aggregation reads normalized case dictionaries without making network calls. Human references remain validation-only and are evaluated separately for Mock, Qwen, and Hybrid outputs.

**Tech Stack:** Python 3.10+ standard library, `unittest`, dataclasses, `urllib.request`, JSON/CSV/Markdown/HTML report writers.

## Global Constraints

- Hybrid metric weight is exactly Qwen 70% and Mock 30%, rounded to two decimals.
- `risk_tags` are the stable-order union of both judges.
- If either judge returns `critical_fail=true`, the fused result is critical and its weighted score is capped at 59.
- Empty Qwen `improvement` uses a deterministic local fallback and does not trigger a retry.
- Recoverable response-contract errors, HTTP 429, and HTTP 5xx receive at most one retry; HTTP 401, 403, and 404 do not retry.
- A 20-case run makes 20 Qwen requests normally and no more than 40 requests under retry.
- Shell environment variables override `--env-file`, which overrides the current-directory `.env`, which overrides code defaults.
- Only `QWEN_API_KEY`, `QWEN_BASE_URL`, and `QWEN_MODEL` may be loaded from dotenv files; values are never executed or expanded.
- API keys must never appear in logs, reports, exceptions, fixtures, commits, or screenshots.
- Existing `--judge mock` output remains backward compatible.
- Tests and CI do not make real Qwen requests.

---

### Task 1: Safe dotenv configuration

**Files:**
- Create: `src/reply_eval/env.py`
- Modify: `src/reply_eval/qwen_judge.py`
- Modify: `src/reply_eval/cli.py`
- Test: `tests/test_env.py`
- Test: `tests/test_qwen_judge.py`

**Interfaces:**
- Produces: `load_qwen_env(explicit_path: Path | None, cwd: Path | None = None) -> Path | None`.
- Produces: `QwenConfig.from_env() -> QwenConfig` after dotenv loading has populated only missing environment variables.
- Produces: CLI option `--env-file PATH`.

- [ ] **Step 1: Write failing dotenv tests**

```python
class EnvTests(unittest.TestCase):
    def test_explicit_file_loads_only_qwen_keys_without_expansion(self):
        path = self.write_env(
            'QWEN_API_KEY="file-key"\nQWEN_MODEL=qwen-plus\nOTHER=blocked\n'
            'QWEN_BASE_URL="https://example.test/$TOKEN"\n'
        )
        with patch.dict(os.environ, {}, clear=True):
            loaded = load_qwen_env(path, path.parent)
            self.assertEqual(loaded, path)
            self.assertEqual(os.environ["QWEN_API_KEY"], "file-key")
            self.assertEqual(os.environ["QWEN_BASE_URL"], "https://example.test/$TOKEN")
            self.assertNotIn("OTHER", os.environ)

    def test_shell_values_override_explicit_and_default_files(self):
        explicit = self.write_env("QWEN_MODEL=file-model\n")
        with patch.dict(os.environ, {"QWEN_MODEL": "shell-model"}, clear=True):
            load_qwen_env(explicit, explicit.parent)
            self.assertEqual(os.environ["QWEN_MODEL"], "shell-model")
```

- [ ] **Step 2: Run the dotenv tests and verify the missing-module failure**

Run: `PYTHONPATH=src python3 -m unittest tests.test_env -v`

Expected: FAIL because `reply_eval.env` does not exist.

- [ ] **Step 3: Implement the minimal safe loader**

```python
ALLOWED_QWEN_KEYS = {"QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL"}

def load_qwen_env(explicit_path: Path | None, cwd: Path | None = None) -> Path | None:
    candidate = explicit_path or (cwd or Path.cwd()) / ".env"
    if not candidate.exists():
        if explicit_path is not None:
            raise EnvFileError(f"env file not found: {candidate}")
        return None
    for line_number, raw_line in enumerate(candidate.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise EnvFileError(f"invalid env line {line_number} in {candidate}")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in ALLOWED_QWEN_KEYS:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return candidate
```

- [ ] **Step 4: Add `--env-file` and load it only for Qwen-backed modes**

```python
parser.add_argument("--env-file", type=Path, help="Qwen 配置文件，shell 环境变量优先")

if args.judge in {"qwen", "hybrid"}:
    load_qwen_env(args.env_file)
```

- [ ] **Step 5: Run focused and existing configuration tests**

Run: `PYTHONPATH=src python3 -m unittest tests.test_env tests.test_qwen_judge -v`

Expected: PASS with no network access.

- [ ] **Step 6: Commit the configuration slice**

```bash
git add src/reply_eval/env.py src/reply_eval/qwen_judge.py src/reply_eval/cli.py tests/test_env.py tests/test_qwen_judge.py
git commit -m "feat: load qwen settings from safe dotenv files"
```

---

### Task 2: Qwen response recovery, fallback, and diagnostics

**Files:**
- Modify: `src/reply_eval/models.py`
- Modify: `src/reply_eval/qwen_judge.py`
- Test: `tests/test_qwen_judge.py`

**Interfaces:**
- `QwenJudge.evaluate(case, knowledge=None) -> CaseEvaluation` remains the public API.
- `CaseEvaluation.evaluator` accepts JSON-serializable metadata values.
- Qwen evaluator metadata includes `request_count`, `retry_count`, and `local_improvement_fallback`.
- `QwenError` messages include case ID and attempt count where available, but never the API key.

- [ ] **Step 1: Add a failing regression test for empty improvement**

```python
def test_empty_improvement_uses_local_fallback_without_retry(self):
    payload = json.loads(VALID_CONTENT)
    payload["improvement"] = ""
    calls = []
    def transport(request, timeout):
        calls.append(request)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}
    result = QwenJudge(self.config, transport=transport).evaluate(self.case)
    self.assertEqual(len(calls), 1)
    self.assertTrue(result.improvement)
    self.assertTrue(result.evaluator["local_improvement_fallback"])
```

- [ ] **Step 2: Run the regression test and verify it fails with the reported defect**

Run: `PYTHONPATH=src python3 -m unittest tests.test_qwen_judge.QwenJudgeTests.test_empty_improvement_uses_local_fallback_without_retry -v`

Expected: FAIL with `improvement must be non-empty text`.

- [ ] **Step 3: Make improvement fallback independent from strict metric validation**

```python
def _fallback_improvement(metrics: dict[str, MetricScore]) -> str:
    lowest = min(WEIGHTS, key=lambda name: (metrics[name].score, list(WEIGHTS).index(name)))
    return IMPROVEMENT_TEMPLATES[lowest]

improvement = payload.get("improvement")
used_fallback = not isinstance(improvement, str) or not improvement.strip()
if used_fallback:
    improvement = _fallback_improvement(metrics)
```

Update `SYSTEM_PROMPT` to use non-empty examples and explicitly require non-empty `reason`, `evidence`, and `improvement` text.

- [ ] **Step 4: Verify the empty-improvement test passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_qwen_judge.QwenJudgeTests.test_empty_improvement_uses_local_fallback_without_retry -v`

Expected: PASS and exactly one transport call.

- [ ] **Step 5: Add failing tests for retry classes and sanitized diagnostics**

```python
def test_invalid_json_retries_once_then_succeeds(self):
    responses = iter([
        {"choices": [{"message": {"content": "not json"}}]},
        {"choices": [{"message": {"content": VALID_CONTENT}}]},
    ])
    result = QwenJudge(self.config, transport=lambda request, timeout: next(responses)).evaluate(self.case)
    self.assertEqual(result.evaluator["request_count"], 2)
    self.assertEqual(result.evaluator["retry_count"], 1)

def test_http_404_does_not_retry_and_names_model_without_secret(self):
    calls = 0
    def transport(request, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(request.full_url, 404, "missing", {}, None)
    with self.assertRaisesRegex(QwenError, "QWEN_MODEL.*qwen-test") as caught:
        QwenJudge(self.config, transport=transport).evaluate(self.case)
    self.assertEqual(calls, 1)
    self.assertNotIn(self.config.api_key, str(caught.exception))
```

Cover HTTP 401/403 without retry, HTTP 429/5xx with one retry, malformed/partial JSON with one retry, and retry exhaustion with case ID and `attempts=2`.

- [ ] **Step 6: Run retry tests and verify missing retry behavior**

Run: `PYTHONPATH=src python3 -m unittest tests.test_qwen_judge -v`

Expected: FAIL because the current judge makes one attempt and emits generic HTTP messages.

- [ ] **Step 7: Implement a two-attempt request loop**

```python
for attempt in range(1, 3):
    try:
        response = self.transport(self._request(user_payload, repair_error), self.config.timeout)
        parsed = self._response_payload(response)
        return self._parse(case["id"], parsed, request_count=attempt)
    except HTTPError as exc:
        if exc.code not in {429} and not 500 <= exc.code <= 599:
            self._raise_http(case["id"], exc.code, attempt)
        last_error = f"HTTP {exc.code}"
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        last_error = str(exc)
    if attempt == 2:
        self._raise_sanitized(
            f"Qwen failed for case {case['id']} after attempts=2: {last_error}"
        )
    repair_error = last_error
```

The second request adds only a short contract-repair instruction and the sanitized validation error. Network timeout and non-HTTP URL errors remain explicit failures without automatic retry.

- [ ] **Step 8: Run all Qwen tests**

Run: `PYTHONPATH=src python3 -m unittest tests.test_qwen_judge -v`

Expected: PASS; request counts match the retry policy and no assertion output contains a secret.

- [ ] **Step 9: Commit the resilient Qwen judge**

```bash
git add src/reply_eval/models.py src/reply_eval/qwen_judge.py tests/test_qwen_judge.py
git commit -m "fix: make qwen response handling resilient"
```

---

### Task 3: Deterministic hybrid fusion

**Files:**
- Create: `src/reply_eval/hybrid_judge.py`
- Create: `tests/test_hybrid_judge.py`
- Modify: `src/reply_eval/models.py`

**Interfaces:**
- Produces: `HybridJudge(mock_judge: MockJudge, qwen_judge: QwenJudge)`.
- Produces: `HybridJudge.evaluate(case, knowledge=None) -> HybridEvaluation`.
- `HybridEvaluation.final` is the fused `CaseEvaluation`; `.mock` and `.qwen` retain component evaluations; `.to_case_dict()` serializes all audit fields.

- [ ] **Step 1: Write failing fusion tests**

```python
def test_fuses_each_metric_seventy_thirty(self):
    mock = evaluation(50, tags=["mock-risk"])
    qwen = evaluation(100, tags=["qwen-risk"])
    result = HybridJudge(StaticJudge(mock), StaticJudge(qwen)).evaluate(CASE)
    self.assertEqual(result.final.metrics["usefulness"].score, 85.0)
    self.assertEqual(result.final.risk_tags, ["mock-risk", "qwen-risk"])
    self.assertEqual(result.max_disagreement, 50.0)

def test_critical_failure_from_either_judge_is_preserved(self):
    result = HybridJudge(
        StaticJudge(evaluation(100, critical=True)),
        StaticJudge(evaluation(100)),
    ).evaluate(CASE)
    self.assertTrue(result.final.critical_fail)
    self.assertEqual(result.final.overall_score, 59.0)
```

- [ ] **Step 2: Run fusion tests and verify the missing-module failure**

Run: `PYTHONPATH=src python3 -m unittest tests.test_hybrid_judge -v`

Expected: FAIL because `reply_eval.hybrid_judge` does not exist.

- [ ] **Step 3: Implement fusion and auditable serialization**

```python
score = round(qwen_metric.score * 0.70 + mock_metric.score * 0.30, 2)
reason = f"Qwen：{qwen_metric.reason}；Mock：{mock_metric.reason}"
evidence = [f"Qwen: {item}" for item in qwen_metric.evidence] + [
    f"Mock: {item}" for item in mock_metric.evidence
]
```

Serialize `mock_metrics`, `qwen_metrics`, fused `metrics`, per-metric absolute differences, `max_disagreement`, component overall scores, request/retry counts, and fallback status. Preserve risk-tag order using `list(dict.fromkeys(...))`.

- [ ] **Step 4: Run hybrid tests and the model contract tests**

Run: `PYTHONPATH=src python3 -m unittest tests.test_hybrid_judge tests.test_models -v`

Expected: PASS with exact 70/30 scores and critical cap behavior.

- [ ] **Step 5: Commit the fusion layer**

```bash
git add src/reply_eval/hybrid_judge.py src/reply_eval/models.py tests/test_hybrid_judge.py
git commit -m "feat: add auditable hybrid judge fusion"
```

---

### Task 4: CLI orchestration and three-way validation

**Files:**
- Modify: `src/reply_eval/cli.py`
- Modify: `src/reply_eval/validation.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_validation.py`

**Interfaces:**
- CLI accepts `--judge hybrid`.
- Produces metadata keys `judge_mode`, `qwen_model`, `request_count`, `retry_count`, and `local_improvement_fallback_count` for Qwen-backed runs.
- Hybrid validation is `{mock: ..., qwen: ..., hybrid: ...}` using the same human references and separate component evaluations.

- [ ] **Step 1: Add a failing hybrid CLI test with injected local judges**

Refactor `main` to accept an internal-only factory parameter without changing console usage:

```python
def main(argv: list[str] | None = None, judge_factory: JudgeFactory | None = None) -> int:
    ...
```

Test with deterministic in-process judges and assert:

```python
self.assertEqual(result["metadata"]["judge_mode"], "hybrid")
self.assertEqual(len(result["cases"]), 20)
self.assertIn("mock_metrics", result["cases"][0])
self.assertIn("qwen_metrics", result["cases"][0])
self.assertEqual(set(result["validation"]), {"mock", "qwen", "hybrid"})
```

- [ ] **Step 2: Run CLI and validation tests and verify failure**

Run: `PYTHONPATH=src python3 -m unittest tests.test_cli tests.test_validation -v`

Expected: FAIL because `hybrid` is not an accepted choice and validation is single-run only.

- [ ] **Step 3: Implement mode construction and component collection**

```python
if args.judge == "mock":
    judge = MockJudge()
elif args.judge == "qwen":
    judge = QwenJudge(QwenConfig.from_env())
else:
    judge = HybridJudge(MockJudge(), QwenJudge(QwenConfig.from_env()))
```

For hybrid results, append `result.final` to final evaluations, preserve `result.mock` and `result.qwen` for validation, and serialize through `result.to_case_dict()`. For other modes, preserve the existing case dictionary shape.

- [ ] **Step 4: Add separate component validation**

```python
validation = {
    "mock": build_validation(references, mock_evaluations),
    "qwen": build_validation(references, qwen_evaluations),
    "hybrid": build_validation(references, final_evaluations),
}
```

Ensure human reference text is never added to judge input and is processed only after all scores are produced.

- [ ] **Step 5: Run CLI, validation, and Qwen tests**

Run: `PYTHONPATH=src python3 -m unittest tests.test_cli tests.test_validation tests.test_qwen_judge -v`

Expected: PASS; hybrid tests use zero real HTTP requests.

- [ ] **Step 6: Commit orchestration**

```bash
git add src/reply_eval/cli.py src/reply_eval/validation.py tests/test_cli.py tests/test_validation.py
git commit -m "feat: run hybrid evaluation from the cli"
```

---

### Task 5: Hybrid report summaries and disagreement review

**Files:**
- Modify: `src/reply_eval/reporting.py`
- Modify: `tests/test_reporting.py`

**Interfaces:**
- `build_summary(cases)` retains existing summary keys.
- Hybrid summary additionally contains `component_metric_means`, `component_overall_means`, `largest_disagreements`, `request_count`, `retry_count`, and `local_improvement_fallback_count`.
- JSON, CSV, Markdown, and HTML all distinguish final Hybrid values from component values.

- [ ] **Step 1: Add failing report tests for three score sets and disagreement ordering**

```python
self.assertEqual(payload["summary"]["largest_disagreements"][0]["id"], "case_03")
self.assertEqual(
    set(payload["summary"]["component_overall_means"]),
    {"mock", "qwen", "hybrid"},
)
self.assertIn("Mock / Qwen / Hybrid", markdown)
self.assertIn("最大分歧", html)
self.assertIn("mock_overall_score", csv_header)
self.assertIn("qwen_overall_score", csv_header)
```

- [ ] **Step 2: Run report tests and verify missing hybrid fields**

Run: `PYTHONPATH=src python3 -m unittest tests.test_reporting -v`

Expected: FAIL because current reports only summarize `metrics` and final overall scores.

- [ ] **Step 3: Extend aggregation without changing mock summaries**

```python
hybrid_cases = [case for case in cases if "mock_metrics" in case and "qwen_metrics" in case]
if hybrid_cases:
    summary["component_overall_means"] = {
        mode: round(mean(case[f"{mode}_overall_score"] for case in hybrid_cases), 2)
        for mode in ("mock", "qwen")
    }
    summary["component_overall_means"]["hybrid"] = summary["overall_mean"]
    summary["largest_disagreements"] = sorted(
        ({"id": case["id"], "max_difference": case["judge_disagreement"]["max"]} for case in hybrid_cases),
        key=lambda item: (-item["max_difference"], item["id"]),
    )[:3]
```

Aggregate component metric means and Qwen call metadata from the same case dictionaries.

- [ ] **Step 4: Add compact Hybrid sections to Markdown and HTML and component columns to CSV**

The report must show final score as the headline; label component values as diagnostics. Add a “分歧最大 3 条” section with per-metric differences and no raw prompt/API payloads.

- [ ] **Step 5: Run report and CLI tests**

Run: `PYTHONPATH=src python3 -m unittest tests.test_reporting tests.test_cli -v`

Expected: PASS for both mock-compatible and hybrid fixtures.

- [ ] **Step 6: Commit report support**

```bash
git add src/reply_eval/reporting.py tests/test_reporting.py tests/test_cli.py
git commit -m "feat: report hybrid scores and judge disagreement"
```

---

### Task 6: Documentation, safe examples, and deliverable refresh

**Files:**
- Modify: `.env.example`
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `outputs/evaluation_results.json`
- Modify: `outputs/case_scores.csv`
- Modify: `outputs/evaluation_report.md`
- Modify: `outputs/evaluation_report.html`
- Modify: `screenshots/development_process.png`
- Modify: `screenshots/evaluation_results.png`
- Test: `tests/test_security.py`

**Interfaces:**
- README provides copy-paste commands for mock, qwen, and recommended hybrid modes.
- `.env.example` contains placeholders only.
- Committed sample outputs remain Mock results unless a newly rotated user key is supplied locally; they are explicitly labeled as Mock and are not presented as Qwen results.

- [ ] **Step 1: Add repository safety regression tests**

```python
def test_dotenv_is_ignored(self):
    ignored = subprocess.run(
        ["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True
    )
    self.assertEqual(ignored.returncode, 0)

def test_tracked_text_does_not_contain_qwen_secret_prefix(self):
    secret_pattern = re.compile(r"s" + r"k-[A-Za-z0-9._-]{12,}")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    for relative in tracked:
        path = ROOT / relative
        if path.is_file() and path.suffix not in {".png"}:
            self.assertIsNone(secret_pattern.search(path.read_text(encoding="utf-8", errors="ignore")))
```

- [ ] **Step 2: Run the safety test and inspect its expected state**

Run: `PYTHONPATH=src python3 -m unittest tests.test_security -v`

Expected: PASS if current ignore and tracked files are clean; this test becomes a regression gate before documentation changes.

- [ ] **Step 3: Update configuration examples and README**

Document:

```bash
cp .env.example .env
# Edit .env locally with a newly generated key; never commit it.
PYTHONPATH=src python3 -m reply_eval.cli \
  --input task3_auto_replies.json \
  --human-ref task3_human_ref.json \
  --output-dir outputs-hybrid \
  --judge hybrid
```

Explain the 70/30 rule, retry matrix, empty-improvement fallback, model-ID troubleshooting, cost ceiling, three score sets, disagreement review, limitations, and that the previously exposed key must be revoked rather than reused.

- [ ] **Step 4: Regenerate deterministic Mock deliverables**

Run:

```bash
PYTHONPATH=src python3 -m reply_eval.cli \
  --input task3_auto_replies.json \
  --human-ref task3_human_ref.json \
  --output-dir outputs \
  --judge mock
```

Expected: 20 cases, four output paths, and metadata `judge_mode=mock`.

- [ ] **Step 5: Capture updated development and result screenshots**

Capture one terminal/IDE view showing the test run and relevant project files, and one browser/terminal view showing the generated evaluation report. Ensure neither screenshot contains `.env`, shell history, API keys, access tokens, or private account details.

- [ ] **Step 6: Run the complete verification suite**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m reply_eval.cli --input task3_auto_replies.json --human-ref task3_human_ref.json --output-dir /tmp/reply-eval-final-check --judge mock
git diff --check
git grep -nE 's[k]-[A-Za-z0-9._-]{12,}' -- . ':!screenshots/*.png'
```

Expected: all tests PASS, mock smoke run evaluates 20 cases, diff check is empty, and secret scan returns no matches.

- [ ] **Step 7: Commit documentation and reproducible deliverables**

```bash
git add .env.example .gitignore README.md outputs screenshots tests/test_security.py
git commit -m "docs: explain and verify hybrid evaluation workflow"
```

---

### Task 7: Review, final verification, and GitHub publication

**Files:**
- Review all files changed since commit `d4d7d62`.

**Interfaces:**
- Produces a reviewed commit range with no Critical or Important findings.
- Updates public branch `origin/main` only after fresh verification.

- [ ] **Step 1: Review requirements against the approved design**

Check every section of `docs/superpowers/specs/2026-09-21-hybrid-qwen-evaluator-design.md` against tests and rendered outputs. Confirm the live API remains unused during automated verification.

- [ ] **Step 2: Request a focused code review**

Use `requesting-code-review` with base SHA `d4d7d62`, current HEAD, the approved design, and this plan. Fix every Critical and Important issue using a new red-green test cycle.

- [ ] **Step 3: Run fresh final verification after review fixes**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m reply_eval.cli --input task3_auto_replies.json --human-ref task3_human_ref.json --output-dir /tmp/reply-eval-final-check --judge mock
git diff --check
git status --short
git grep -nE 's[k]-[A-Za-z0-9._-]{12,}' -- . ':!screenshots/*.png'
```

Expected: all tests PASS; mock smoke run reports 20 cases; no whitespace or secret matches; only explicitly preserved user-generated timestamp changes remain uncommitted, if any.

- [ ] **Step 4: Push reviewed commits to the public repository**

Run: `git push origin HEAD:main`

Expected: GitHub accepts the update and the public repository resolves at `https://github.com/CuitSuanNaiGaiGai/auto-reply-quality-evaluation`.
