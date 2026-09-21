"""Command-line entry point for complete evaluation runs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .env import EnvFileError, load_qwen_env
from .io import InputError, load_cases, load_human_references
from .mock_judge import MockJudge
from .models import RunResult
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
        "--judge", choices=("mock", "qwen"), default="mock", help="评估后端"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Qwen 配置文件；已存在的 shell 环境变量优先",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cases = load_cases(args.input)
        references = None
        if args.human_ref:
            references = load_human_references(
                args.human_ref, {case["id"] for case in cases}
            )
        if args.judge == "mock":
            judge = MockJudge()
        else:
            load_qwen_env(args.env_file)
            judge = QwenJudge(QwenConfig.from_env())
    except (EnvFileError, InputError, QwenError) as exc:
        print(f"configuration/input error: {exc}", file=sys.stderr)
        return 2

    evaluations = []
    rendered_cases = []
    try:
        for case in cases:
            evaluation = judge.evaluate(case)
            evaluations.append(evaluation)
            rendered = evaluation.to_dict()
            rendered["user_question"] = case["user_question"]
            rendered["auto_reply"] = case["auto_reply"]
            rendered_cases.append(rendered)
    except QwenError as exc:
        print(f"judge error: {exc}", file=sys.stderr)
        return 3

    validation = (
        build_validation(references, evaluations) if references is not None else {}
    )
    result = RunResult(
        metadata={
            "judge_mode": args.judge,
            "evaluator": evaluations[0].evaluator if evaluations else {},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_file": args.input.name,
            "human_reference_used_for_scoring": False,
        },
        summary=build_summary(rendered_cases),
        validation=validation,
        cases=rendered_cases,
    )
    try:
        paths = write_reports(result, args.output_dir)
    except OSError as exc:
        print(f"report error: {exc}", file=sys.stderr)
        return 4

    print(f"已评估 {len(cases)} 条回复（judge={args.judge}）")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
