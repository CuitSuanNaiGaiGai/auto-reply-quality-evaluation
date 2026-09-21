"""Directional validation against narrative human annotations."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

from .models import CaseEvaluation


POSITIVE_PHRASES = (
    "处理得不错",
    "基本正确",
    "质量尚可",
    "基本合理",
    "也有价值",
    "是可接受的",
)
NEGATIVE_PHRASES = (
    "答非所问",
    "正确但没用",
    "把责任推给",
    "没有体现主动",
    "没有帮用户",
    "没有直接给出解决方案",
    "增加了用户的操作负担",
    "把用户推走",
    "只是泛泛",
)

ISSUE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        r"把责任推给|让用户自己|把用户推走|没有帮用户实际",
        ("self_service_deflection", "specific_case_unresolved"),
    ),
    (r"没有追问|需要追问|需要确认", ("missing_clarification",)),
    (r"情绪安抚不够|语气和力度不够", ("emotion_underaddressed",)),
    (r"两个问题|同时处理多个问题", ("multi_intent_missed",)),
)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman_rank(left: Iterable[float], right: Iterable[float]) -> float | None:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError("rank inputs must have equal length")
    if len(left_values) < 2:
        return None
    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    left_mean = mean(left_ranks)
    right_mean = mean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(
        sum((value - right_mean) ** 2 for value in right_ranks)
    )
    if left_scale == 0 or right_scale == 0:
        return None
    return round(numerator / (left_scale * right_scale), 4)


def summarize_tiers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["tier"])].append(float(row["score"]))
    averages = {
        tier: round(mean(scores), 2) for tier, scores in sorted(grouped.items())
    }
    gap = None
    if grouped.get("positive") and grouped.get("negative"):
        gap = round(mean(grouped["positive"]) - mean(grouped["negative"]), 2)
    return {"tier_means": averages, "positive_negative_gap": gap}


def classify_note(note: str) -> tuple[str, str]:
    for phrase in NEGATIVE_PHRASES:
        if phrase in note:
            return "negative", phrase
    for phrase in POSITIVE_PHRASES:
        if phrase in note:
            return "positive", phrase
    return "middle", "未命中明确正向或严重负向短语"


def _expected_tags(note: str) -> list[str]:
    tags: list[str] = []
    for pattern, candidates in ISSUE_RULES:
        if re.search(pattern, note):
            tags.extend(candidates)
    return list(dict.fromkeys(tags))


def build_validation(
    references: list[dict[str, str]], evaluations: list[CaseEvaluation]
) -> dict[str, Any]:
    by_id = {evaluation.case_id: evaluation for evaluation in evaluations}
    rows: list[dict[str, Any]] = []
    matched_issue_cases = 0
    issue_cases = 0
    disagreements: list[dict[str, Any]] = []
    tier_numbers = {"negative": 1.0, "middle": 2.0, "positive": 3.0}

    for reference in references:
        evaluation = by_id[reference["id"]]
        tier, matched_phrase = classify_note(reference["annotator_notes"])
        expected_tags = _expected_tags(reference["annotator_notes"])
        matched_tags = sorted(set(expected_tags) & set(evaluation.risk_tags))
        if expected_tags:
            issue_cases += 1
            if matched_tags:
                matched_issue_cases += 1
        row = {
            "id": reference["id"],
            "tier": tier,
            "tier_value": tier_numbers[tier],
            "matched_phrase": matched_phrase,
            "score": evaluation.overall_score,
            "expected_issue_tags": expected_tags,
            "matched_issue_tags": matched_tags,
        }
        rows.append(row)
        if (tier == "positive" and evaluation.overall_score < 70) or (
            tier == "negative" and evaluation.overall_score >= 70
        ):
            disagreements.append(
                {
                    "id": reference["id"],
                    "human_tier": tier,
                    "automatic_score": evaluation.overall_score,
                    "reason": "人工档位与自动分数阈值不一致",
                }
            )

    tier_summary = summarize_tiers(rows)
    correlation = spearman_rank(
        [row["tier_value"] for row in rows],
        [row["score"] for row in rows],
    )
    issue_match_rate = (
        round(matched_issue_cases / issue_cases, 4) if issue_cases else None
    )
    return {
        "method": "人工参考仅用于三档排序与问题标签验证，未参与评分",
        "sample_size": len(rows),
        **tier_summary,
        "spearman_correlation": correlation,
        "issue_tag_match_rate": issue_match_rate,
        "issue_cases": issue_cases,
        "disagreements": disagreements,
        "rows": rows,
    }
