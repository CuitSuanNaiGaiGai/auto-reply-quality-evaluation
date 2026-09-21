"""Command-line entry point for complete evaluation runs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .env import EnvFileError, load_qwen_env
from .hybrid_judge import HybridEvaluation, HybridJudge
from .io import InputError, load_cases, load_human_references
from .mock_judge import MockJudge
from .models import CaseEvaluation, RunResult
from .qwen_judge import QwenConfig, QwenError, QwenJudge
from .reporting import build_summary, write_reports
from .validation import build_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="评估客服自动回复的意图、服务闭环、事实依据、语气和清晰度。"
    )
    parser.add_argument("--input", type=Path, required=True, help="自动回复 JSON")
    parser.add_argument(
        "--human-ref", type=Path, help="可选人工参考 JSON，仅用于验证"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="报告输出目录"
    )
    parser.add_argument(
        "--judge",
        choices=("mock", "qwen", "hybrid"),
        default="mock",
        help="评估后端；推荐 hybrid",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Qwen 配置文件；已存在的 shell 环境变量优先",
    )
    return parser


JudgeFactory = Callable[[str], Any]


def _default_judge(mode: str, env_file: Path | None) -> Any:
    if mode == "mock":
        return MockJudge()
    load_qwen_env(env_file)
    qwen = QwenJudge(QwenConfig.from_env())
    if mode == "qwen":
        return qwen
    return HybridJudge(MockJudge(), qwen)


def main(
    argv: list[str] | None = None,
    judge_factory: JudgeFactory | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        cases = load_cases(args.input)
        references = None
        if args.human_ref:
            references = load_human_references(
                args.human_ref, {case["id"] for case in cases}
            )
        judge = (
            judge_factory(args.judge)
            if judge_factory is not None
            else _default_judge(args.judge, args.env_file)
        )
    except (EnvFileError, InputError, QwenError) as exc:
        print(f"configuration/input error: {exc}", file=sys.stderr)
        return 2

    evaluations: list[CaseEvaluation] = []
    mock_evaluations: list[CaseEvaluation] = []
    qwen_evaluations: list[CaseEvaluation] = []
    rendered_cases: list[dict[str, Any]] = []
    try:
        for case in cases:
            raw_evaluation = judge.evaluate(case)
            if isinstance(raw_evaluation, HybridEvaluation):
                evaluation = raw_evaluation.final
                mock_evaluations.append(raw_evaluation.mock)
                qwen_evaluations.append(raw_evaluation.qwen)
                rendered = raw_evaluation.to_case_dict()
            else:
                evaluation = raw_evaluation
                rendered = evaluation.to_dict()
            evaluations.append(evaluation)
            rendered["user_question"] = case["user_question"]
            rendered["auto_reply"] = case["auto_reply"]
            rendered_cases.append(rendered)
    except QwenError as exc:
        print(f"judge error: {exc}", file=sys.stderr)
        return 3

    if references is None:
        validation: dict[str, Any] = {}
    elif args.judge == "hybrid":
        validation = {
            "mock": build_validation(references, mock_evaluations),
            "qwen": build_validation(references, qwen_evaluations),
            "hybrid": build_validation(references, evaluations),
        }
    else:
        validation = build_validation(references, evaluations)

    metadata: dict[str, Any] = {
        "judge_mode": args.judge,
        "evaluator": evaluations[0].evaluator if evaluations else {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": args.input.name,
        "human_reference_used_for_scoring": False,
    }
    if args.judge == "qwen":
        metadata.update(
            {
                "qwen_model": evaluations[0].evaluator.get("model"),
                "request_count": sum(
                    int(value.evaluator.get("request_count", 0))
                    for value in evaluations
                ),
                "retry_count": sum(
                    int(value.evaluator.get("retry_count", 0))
                    for value in evaluations
                ),
                "local_improvement_fallback_count": sum(
                    bool(
                        value.evaluator.get(
                            "local_improvement_fallback", False
                        )
                    )
                    for value in evaluations
                ),
            }
        )
    elif args.judge == "hybrid":
        metadata.update(
            {
                "qwen_model": qwen_evaluations[0].evaluator.get("model"),
                "request_count": sum(
                    int(case.get("qwen_request_count", 0))
                    for case in rendered_cases
                ),
                "retry_count": sum(
                    int(case.get("qwen_retry_count", 0))
                    for case in rendered_cases
                ),
                "local_improvement_fallback_count": sum(
                    bool(case.get("local_improvement_fallback", False))
                    for case in rendered_cases
                ),
            }
        )
    result = RunResult(
        metadata=metadata,
        summary=build_summary(rendered_cases),
        validation=validation,
        cases=rendered_cases,
    )
    try:
        paths = write_reports(result, args.output_dir)
    except OSError as exc:
        print(f"report error: {exc}", file=sys.stderr)
        return 4

    mode_details = f"judge={args.judge}"
    if args.judge in {"qwen", "hybrid"}:
        mode_details += (
            f", model={metadata['qwen_model']}, "
            f"requests={metadata['request_count']}"
        )
    print(f"已评估 {len(cases)} 条回复（{mode_details}）")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
