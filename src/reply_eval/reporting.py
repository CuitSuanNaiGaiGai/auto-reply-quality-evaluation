"""Aggregate scores and render all deliverable report formats."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from .models import RunResult, WEIGHTS


METRIC_LABELS = {
    "intent_accuracy": "意图准确性",
    "usefulness": "有用性与服务闭环",
    "groundedness": "事实依据与幻觉风险",
    "tone": "语气与情绪适配",
    "clarity": "清晰与简洁",
}


def _quality_band(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 60:
        return "needs_improvement"
    return "poor"


def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("cannot summarize an empty evaluation run")
    scores = [float(case["overall_score"]) for case in cases]
    metric_stats: dict[str, Any] = {}
    for name in WEIGHTS:
        values = [float(case["metrics"][name]["score"]) for case in cases]
        metric_stats[name] = {
            "label": METRIC_LABELS[name],
            "weight": WEIGHTS[name],
            "mean": round(mean(values), 2),
            "median": round(median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "distribution": {
                "poor_0_59": sum(value < 60 for value in values),
                "acceptable_60_74": sum(60 <= value < 75 for value in values),
                "strong_75_100": sum(value >= 75 for value in values),
            },
        }
    risk_counts = Counter(
        tag for case in cases for tag in case.get("risk_tags", [])
    )
    band_counts = Counter(_quality_band(score) for score in scores)
    worst = sorted(cases, key=lambda case: (case["overall_score"], case["id"]))[:3]
    summary: dict[str, Any] = {
        "case_count": len(cases),
        "overall_mean": round(mean(scores), 2),
        "overall_median": round(median(scores), 2),
        "overall_min": round(min(scores), 2),
        "overall_max": round(max(scores), 2),
        "quality_bands": {
            key: band_counts.get(key, 0)
            for key in ["excellent", "good", "needs_improvement", "poor"]
        },
        "metric_stats": metric_stats,
        "risk_tag_counts": dict(risk_counts.most_common()),
        "critical_fail_count": sum(bool(case["critical_fail"]) for case in cases),
        "worst_three": [case["id"] for case in worst],
    }
    hybrid_cases = [
        case
        for case in cases
        if "mock_metrics" in case and "qwen_metrics" in case
    ]
    if hybrid_cases:
        summary["component_overall_means"] = {
            "mock": round(
                mean(
                    float(case["mock_overall_score"])
                    for case in hybrid_cases
                ),
                2,
            ),
            "qwen": round(
                mean(
                    float(case["qwen_overall_score"])
                    for case in hybrid_cases
                ),
                2,
            ),
            "hybrid": summary["overall_mean"],
        }
        summary["component_metric_means"] = {
            mode: {
                name: round(
                    mean(
                        float(case[f"{mode}_metrics"][name]["score"])
                        for case in hybrid_cases
                    ),
                    2,
                )
                for name in WEIGHTS
            }
            for mode in ("mock", "qwen")
        }
        summary["component_metric_means"]["hybrid"] = {
            name: stats["mean"] for name, stats in metric_stats.items()
        }
        summary["largest_disagreements"] = sorted(
            (
                {
                    "id": case["id"],
                    "max_difference": float(
                        case["judge_disagreement"]["max"]
                    ),
                    "metric_differences": {
                        name: float(case["judge_disagreement"][name])
                        for name in WEIGHTS
                    },
                }
                for case in hybrid_cases
            ),
            key=lambda item: (-item["max_difference"], item["id"]),
        )[:3]
        summary["request_count"] = sum(
            int(case.get("qwen_request_count", 0)) for case in hybrid_cases
        )
        summary["retry_count"] = sum(
            int(case.get("qwen_retry_count", 0)) for case in hybrid_cases
        )
        summary["local_improvement_fallback_count"] = sum(
            bool(case.get("local_improvement_fallback", False))
            for case in hybrid_cases
        )
    return summary


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    cases = payload["cases"]
    by_id = {case["id"]: case for case in cases}
    lines = [
        "# 客服自动回复质量评估报告",
        "",
        f"- 评估样本：{summary['case_count']} 条",
        f"- 整体加权平均分：**{summary['overall_mean']:.2f} / 100**",
        f"- 中位数：{summary['overall_median']:.2f}",
        f"- 分数范围：{summary['overall_min']:.2f} – {summary['overall_max']:.2f}",
        f"- 高风险失败：{summary['critical_fail_count']} 条",
        "",
        "## 指标分布",
        "",
        "| 指标 | 权重 | 均分 | 中位数 | 最低 | 最高 | <60 | 60–74 | ≥75 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in summary["metric_stats"].items():
        distribution = stats["distribution"]
        lines.append(
            f"| {stats['label']} | {stats['weight']:.0%} | {stats['mean']:.2f} | "
            f"{stats['median']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | "
            f"{distribution['poor_0_59']} | {distribution['acceptable_60_74']} | "
            f"{distribution['strong_75_100']} |"
        )
    if "component_overall_means" in summary:
        component_overall = summary["component_overall_means"]
        component_metrics = summary["component_metric_means"]
        lines.extend(
            [
                "",
                "## Hybrid 审计：Mock / Qwen / Hybrid",
                "",
                "最终 Hybrid 分数使用 Qwen 70% + Mock 30%；组件分仅用于诊断。",
                "",
                "| 评估器 | 综合均分 | 意图 | 有用性 | 事实依据 | 语气 | 清晰度 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in ("mock", "qwen", "hybrid"):
            values = component_metrics[mode]
            lines.append(
                f"| {mode.title()} | {component_overall[mode]:.2f} | "
                + " | ".join(f"{values[name]:.2f}" for name in WEIGHTS)
                + " |"
            )
        lines.extend(
            [
                "",
                f"- Qwen 请求次数：{summary['request_count']}",
                f"- 格式/服务重试次数：{summary['retry_count']}",
                f"- 本地建议降级次数：{summary['local_improvement_fallback_count']}",
                "",
                "### 分歧最大 3 条",
                "",
            ]
        )
        for item in summary["largest_disagreements"]:
            lines.append(
                f"- {item['id']}：最大指标差 {item['max_difference']:.2f} 分"
            )
    lines.extend(["", "## 最差 3 条", ""])
    for rank, case_id in enumerate(summary["worst_three"], start=1):
        case = by_id[case_id]
        tags = "、".join(case["risk_tags"]) or "无"
        lines.extend(
            [
                f"### {rank}. {case_id} — {case['overall_score']:.2f} 分",
                "",
                f"- 用户问题：{case['user_question']}",
                f"- 自动回复：{case['auto_reply']}",
                f"- 主要风险：{tags}",
                f"- 改进建议：{case['improvement']}",
                "",
            ]
        )
    validation = payload.get("validation", {})
    lines.extend(["## 人工参考验证", ""])
    if validation and all(
        mode in validation for mode in ("mock", "qwen", "hybrid")
    ):
        lines.extend(
            [
                "| 评估器 | 样本数 | Spearman | 正负档均分差 | 标签匹配率 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for mode in ("mock", "qwen", "hybrid"):
            item = validation[mode]
            match_rate = item.get("issue_tag_match_rate")
            match_text = (
                f"{match_rate:.2%}"
                if isinstance(match_rate, (int, float))
                else "N/A"
            )
            lines.append(
                f"| {mode.title()} | {item.get('sample_size')} | "
                f"{item.get('spearman_correlation')} | "
                f"{item.get('positive_negative_gap')} | {match_text} |"
            )
    elif validation:
        lines.extend(
            [
                f"- 样本数：{validation.get('sample_size')}",
                f"- Spearman 相关系数：{validation.get('spearman_correlation')}",
                f"- 正负档均分差：{validation.get('positive_negative_gap')}",
                f"- 问题标签匹配率：{validation.get('issue_tag_match_rate')}",
                f"- 不一致 case 数：{len(validation.get('disagreements', []))}",
            ]
        )
    else:
        lines.append("未提供人工参考文件。")
    lines.extend(
        [
            "",
            "> 注：“unsupported_claim”表示当前输入无法验证、需要外部知识库核实，不等同于已证明错误。",
            "",
        ]
    )
    return "\n".join(lines)


def _score_color(score: float) -> str:
    if score >= 75:
        return "#24b47e"
    if score >= 60:
        return "#f0a329"
    return "#e45c67"


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    cases = payload["cases"]
    by_id = {case["id"]: case for case in cases}
    metric_cards: list[str] = []
    for stats in summary["metric_stats"].values():
        value = stats["mean"]
        metric_cards.append(
            f"""<article class="metric">
              <div class="metric-head"><span>{html.escape(stats['label'])}</span><strong>{value:.1f}</strong></div>
              <div class="track"><i style="width:{value}%;background:{_score_color(value)}"></i></div>
              <small>权重 {stats['weight']:.0%} · 范围 {stats['min']:.0f}–{stats['max']:.0f}</small>
            </article>"""
        )
    worst_cards: list[str] = []
    for rank, case_id in enumerate(summary["worst_three"], start=1):
        case = by_id[case_id]
        tags = "".join(
            f"<span class='tag'>{html.escape(tag)}</span>" for tag in case["risk_tags"]
        ) or "<span class='tag neutral'>无明确风险标签</span>"
        metric_rows = "".join(
            f"<li><span>{html.escape(METRIC_LABELS[name])}</span><b>{case['metrics'][name]['score']:.0f}</b></li>"
            for name in WEIGHTS
        )
        worst_cards.append(
            f"""<article class="case-card">
              <header><span class="rank">#{rank}</span><div><h3>{html.escape(case_id)}</h3><p>{case['overall_score']:.1f} / 100</p></div></header>
              <div class="case-grid"><div><label>用户问题</label><p>{html.escape(case['user_question'])}</p>
              <label>自动回复</label><p>{html.escape(case['auto_reply'])}</p></div><ul>{metric_rows}</ul></div>
              <div class="tags">{tags}</div><p class="improve"><b>改进：</b>{html.escape(case['improvement'])}</p>
            </article>"""
        )
    risk_rows = "".join(
        f"<tr><td>{html.escape(tag)}</td><td>{count}</td></tr>"
        for tag, count in summary["risk_tag_counts"].items()
    ) or "<tr><td>无</td><td>0</td></tr>"
    component_html = ""
    if "component_overall_means" in summary:
        component_overall = summary["component_overall_means"]
        component_rows = "".join(
            f"<tr><td>{mode.title()}</td><td>{component_overall[mode]:.2f}</td></tr>"
            for mode in ("mock", "qwen", "hybrid")
        )
        disagreement_rows = "".join(
            f"<tr><td>{html.escape(item['id'])}</td><td>{item['max_difference']:.2f}</td></tr>"
            for item in summary["largest_disagreements"]
        )
        component_html = f"""
<section class="section overview"><div class="panel"><div class="section-title"><h2>Mock / Qwen / Hybrid</h2><p>组件诊断分</p></div><table>{component_rows}</table></div>
<div class="panel"><div class="section-title"><h2>最大分歧</h2><p>优先人工复核</p></div><table>{disagreement_rows}</table>
<p class="audit">请求 {summary['request_count']} · 重试 {summary['retry_count']} · 本地建议降级 {summary['local_improvement_fallback_count']}</p></div></section>"""
    validation = payload.get("validation", {})
    displayed_validation = (
        validation.get("hybrid", {})
        if isinstance(validation, dict) and "hybrid" in validation
        else validation
    )
    correlation = displayed_validation.get("spearman_correlation", "N/A")
    gap = displayed_validation.get("positive_negative_gap", "N/A")
    match_rate = displayed_validation.get("issue_tag_match_rate")
    match_text = f"{match_rate:.1%}" if isinstance(match_rate, (int, float)) else "N/A"
    mode = html.escape(str(payload.get("metadata", {}).get("judge_mode", "unknown")))
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>自动回复质量评估</title>
<style>
:root{{--ink:#172033;--muted:#657085;--paper:#f5f7fb;--card:#fff;--blue:#4b67d1;--line:#e4e8f0}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}
.shell{{max-width:1160px;margin:auto;padding:38px 28px 70px}}.hero{{background:linear-gradient(135deg,#152346,#384f9e);color:white;border-radius:24px;padding:34px;display:grid;grid-template-columns:1fr auto;gap:30px;box-shadow:0 18px 50px #20346a30}}
.eyebrow{{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:#b9c9ff}}h1{{margin:8px 0;font-size:32px}}.hero p{{color:#d7e0ff;margin:0;max-width:640px;line-height:1.65}}
.score{{width:150px;height:150px;border:10px solid #ffffff2b;border-top-color:#7ee0b6;border-radius:50%;display:grid;place-content:center;text-align:center}}.score strong{{font-size:38px}}.score span{{font-size:12px;color:#cad6ff}}
.section{{margin-top:34px}}.section-title{{display:flex;justify-content:space-between;align-items:end;margin-bottom:14px}}h2{{font-size:20px;margin:0}}.section-title p{{margin:0;color:var(--muted);font-size:13px}}
.metrics{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}.metric,.panel,.case-card{{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 4px 18px #1720330a}}.metric{{padding:18px}}.metric-head{{display:flex;justify-content:space-between;gap:15px}}.metric-head strong{{font-size:21px}}.track{{height:8px;background:#eef1f6;border-radius:99px;margin:13px 0 9px;overflow:hidden}}.track i{{display:block;height:100%;border-radius:99px}}small{{color:var(--muted)}}
.overview{{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}}.panel{{padding:20px}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.stat{{background:#f7f8fc;padding:15px;border-radius:12px}}.stat b{{display:block;font-size:24px}}.stat span{{font-size:12px;color:var(--muted)}}table{{width:100%;border-collapse:collapse}}td{{padding:9px;border-bottom:1px solid var(--line);font-size:13px}}td:last-child{{text-align:right;font-weight:700}}
.case-card{{padding:22px;margin-bottom:14px}}.case-card header{{display:flex;align-items:center;gap:12px}}.case-card h3,.case-card header p{{margin:0}}.case-card header p{{color:#e45c67;font-weight:700}}.rank{{background:#edf0ff;color:var(--blue);font-weight:800;border-radius:10px;padding:10px}}.case-grid{{display:grid;grid-template-columns:1.6fr 1fr;gap:24px;margin-top:16px}}label{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}}.case-grid p{{line-height:1.6;margin:5px 0 14px}}ul{{list-style:none;padding:0;margin:0}}li{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding:8px 0;font-size:13px}}.tag{{display:inline-block;background:#fff1f2;color:#ad3c49;padding:5px 8px;border-radius:7px;margin:2px 5px 2px 0;font-size:11px}}.tag.neutral{{background:#edf7f3;color:#27775d}}.improve{{background:#f5f7ff;border-left:3px solid var(--blue);padding:12px;line-height:1.55}}
.foot{{margin-top:30px;color:var(--muted);font-size:12px;line-height:1.7}}@media(max-width:760px){{.hero,.case-grid,.overview{{grid-template-columns:1fr}}.score{{width:110px;height:110px}}.metrics{{grid-template-columns:1fr}}}}
</style></head><body><main class="shell">
<section class="hero"><div><div class="eyebrow">QUALITY EVALUATION · {mode}</div><h1>客服自动回复质量评估</h1><p>{summary['case_count']} 条回复的可解释离线评估。分数衡量意图、服务闭环、事实依据、语气与清晰度；无证据的声明被标为待核实，不直接视为错误。</p></div><div class="score"><strong>{summary['overall_mean']:.1f}</strong><span>OVERALL / 100</span></div></section>
	<section class="section"><div class="section-title"><h2>指标表现</h2><p>均分 · 权重 · 样本范围</p></div><div class="metrics">{''.join(metric_cards)}</div></section>
	{component_html}
	<section class="section overview"><div class="panel"><div class="section-title"><h2>验证摘要</h2></div><div class="stats"><div class="stat"><b>{correlation}</b><span>Spearman 相关</span></div><div class="stat"><b>{gap}</b><span>正负档均分差</span></div><div class="stat"><b>{match_text}</b><span>问题标签匹配率</span></div></div></div><div class="panel"><div class="section-title"><h2>高频风险</h2></div><table>{risk_rows}</table></div></section>
<section class="section"><div class="section-title"><h2>最差 3 条</h2><p>按综合分升序，同分按 ID</p></div>{''.join(worst_cards)}</section>
<p class="foot">报告中 unsupported_claim 仅表示当前输入无法验证，需要商品库、订单系统或政策知识库核实。本报告不将人工参考答案用于单条评分。</p>
</main></body></html>"""


def write_reports(result: RunResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["summary"] = build_summary(payload["cases"])
    result_path = output_dir / "evaluation_results.json"
    csv_path = output_dir / "case_scores.csv"
    markdown_path = output_dir / "evaluation_report.md"
    html_path = output_dir / "evaluation_report.html"

    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        is_hybrid = any("mock_metrics" in case for case in payload["cases"])
        component_fields = (
            [
                "mock_overall_score",
                "qwen_overall_score",
                "max_disagreement",
                "qwen_request_count",
                "qwen_retry_count",
                "local_improvement_fallback",
            ]
            if is_hybrid
            else []
        )
        fieldnames = [
            "id",
            "overall_score",
            *component_fields,
            *WEIGHTS.keys(),
            "risk_tags",
            "critical_fail",
            "improvement",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for case in payload["cases"]:
            row = {
                    "id": case["id"],
                    "overall_score": case["overall_score"],
                    **{
                        name: case["metrics"][name]["score"] for name in WEIGHTS
                    },
                    "risk_tags": "|".join(case["risk_tags"]),
                    "critical_fail": case["critical_fail"],
                    "improvement": case["improvement"],
                }
            if is_hybrid:
                row.update(
                    {
                        "mock_overall_score": case["mock_overall_score"],
                        "qwen_overall_score": case["qwen_overall_score"],
                        "max_disagreement": case["judge_disagreement"]["max"],
                        "qwen_request_count": case["qwen_request_count"],
                        "qwen_retry_count": case["qwen_retry_count"],
                        "local_improvement_fallback": case[
                            "local_improvement_fallback"
                        ],
                    }
                )
            writer.writerow(row)
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    html_path.write_text(_html_report(payload), encoding="utf-8")
    return {
        "json": result_path,
        "csv": csv_path,
        "markdown": markdown_path,
        "html": html_path,
    }
