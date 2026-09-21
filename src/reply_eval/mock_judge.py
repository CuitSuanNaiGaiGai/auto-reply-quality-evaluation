"""Deterministic, explainable fallback judge.

The rules intentionally operate on generic language signals rather than case IDs.
They are a reproducible baseline, not a replacement for product knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import CaseEvaluation, MetricScore


def _find(text: str, patterns: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        found = re.search(pattern, text)
        if found:
            matches.append(found.group(0))
    return matches


def _bounded(score: float) -> float:
    return float(max(0, min(100, round(score))))


SPECIFIC_PATTERNS = (
    r"我的",
    r"我买的",
    r"我那个",
    r"我上次",
    r"这个",
    r"那个",
    r"刚下单",
    r"才买",
    r"收到",
    r"用了[^\uff0c\u3002！？\s]*",
)
NEEDS_CONTEXT_PATTERNS = (
    r"退款",
    r"退货",
    r"快递",
    r"物流",
    r"优惠券",
    r"账号",
    r"补货",
    r"商品",
    r"手机",
    r"耳机",
    r"面膜",
    r"机器人",
)
CLARIFY_PATTERNS = (
    r"请提供",
    r"提供[^\uff0c\u3002！？]*号",
    r"请告诉",
    r"请问",
    r"哪款",
    r"具体[^\uff0c\u3002！？]*",
    r"卡在哪",
    r"倾向哪",
)
PROACTIVE_PATTERNS = (
    r"我帮您",
    r"帮您查",
    r"帮您确认",
    r"我来帮",
    r"马上帮",
    r"会协助您处理",
    r"我们会协助",
)
DEFLECTION_PATTERNS = (
    r"查看[^\uff0c\u3002！？]*(?:详情页|页面|记录|进度|参数|评价)",
    r"联系快递",
    r"联系品牌",
    r"联系客服",
    r"咨询具体航空公司",
    r"耐心等待",
)
ACTION_PATTERNS = (
    r"不要点击",
    r"修改密码",
    r"开启二次验证",
    r"申请退货",
    r"退货退款",
    r"换新",
    r"在[^\uff0c\u3002！？]*页[^\uff0c\u3002！？]*(?:点击|操作|提交)",
    r"[1-9][.\u3001]",
)
EMOTION_PATTERNS = (
    r"态度太差",
    r"没人理",
    r"久等",
    r"又是坏",
    r"连续",
    r"太复杂",
    r"搞半天",
    r"不知道怎么",
    r"异地登录",
    r"皮肤敏感",
)
EMPATHY_PATTERNS = (
    r"抱歉",
    r"对不起",
    r"理解您",
    r"给您带来",
    r"确实不应该",
    r"非常重视",
    r"不用再等",
)
CLAIM_PATTERNS = (
    r"\d+\s*(?:-|~|–|至)\s*\d+\s*(?:个)?(?:工作日|天|小时|Wh|mAh)",
    r"不超过\s*\d+\s*Wh",
    r"\d+天质保",
    r"根据[^\uff0c\u3002！？]*规定",
    r"运费由[^\uff0c\u3002！？]*承担",
    r"采用的是[^\uff0c\u3002！？]*材质",
    r"属于质量问题",
    r"可以申请[^\uff0c\u3002！？]*优惠券",
    r"系统会自动提醒",
)
CAPABILITY_PATTERNS = (
    r"我们会将[^\uff0c\u3002！？]*转达",
    r"我们会将[^\uff0c\u3002！？]*反馈",
    r"我们会加强",
    r"系统会自动提醒",
    r"可以申请额外",
)
COMPLETED_ACTION_PATTERNS = (
    r"已为您(?:退款|补偿|冻结|设置|申请)",
    r"已经为您(?:退款|补偿|冻结|设置|申请)",
)


@dataclass(frozen=True)
class Signals:
    specific: list[str]
    context_topic: list[str]
    clarification: list[str]
    proactive: list[str]
    deflection: list[str]
    actions: list[str]
    emotion: list[str]
    empathy: list[str]
    claims: list[str]
    capabilities: list[str]
    completed_actions: list[str]
    multi_intent: bool
    covered_intents: int


def _signals(question: str, reply: str) -> Signals:
    multi_intent = bool(re.search(r"顺便|两个问题|分别|又", question))
    intent_terms = {
        "return": ("退货" in question, "退货" in reply or "换货" in reply),
        "refund": ("退款" in question, "退款" in reply),
        "shipping": (
            "快递" in question or "物流" in question,
            "快递" in reply or "物流" in reply,
        ),
    }
    covered = sum(1 for asked, answered in intent_terms.values() if asked and answered)
    return Signals(
        specific=_find(question, SPECIFIC_PATTERNS),
        context_topic=_find(question, NEEDS_CONTEXT_PATTERNS),
        clarification=_find(reply, CLARIFY_PATTERNS),
        proactive=_find(reply, PROACTIVE_PATTERNS),
        deflection=_find(reply, DEFLECTION_PATTERNS),
        actions=_find(reply, ACTION_PATTERNS),
        emotion=_find(question, EMOTION_PATTERNS),
        empathy=_find(reply, EMPATHY_PATTERNS),
        claims=_find(reply, CLAIM_PATTERNS),
        capabilities=_find(reply, CAPABILITY_PATTERNS),
        completed_actions=_find(reply, COMPLETED_ACTION_PATTERNS),
        multi_intent=multi_intent,
        covered_intents=covered,
    )


def _evidence(label: str, matches: list[str]) -> list[str]:
    if matches:
        return [f"{label}: {match}" for match in matches[:3]]
    return ["未检测到相关文本信号"]


class MockJudge:
    """Score replies from visible, deterministic language signals."""

    version = "mock-rubric-1.0"

    def evaluate(
        self, case: dict[str, str], knowledge: str | None = None
    ) -> CaseEvaluation:
        question = case["user_question"]
        reply = case["auto_reply"]
        signal = _signals(question, reply)
        tags: list[str] = []

        intent = 75.0
        if signal.clarification:
            intent += 15
        if signal.specific and signal.context_topic and not signal.clarification:
            intent -= 20
            tags.extend(["specific_case_unresolved", "missing_clarification"])
        if signal.multi_intent:
            if signal.covered_intents >= 2 or "分别" in reply:
                intent += 10
            else:
                intent -= 25
                tags.append("multi_intent_missed")

        usefulness = 60.0
        if signal.proactive:
            usefulness += 25
        if signal.clarification:
            usefulness += 15
        if signal.actions:
            usefulness += 10
        if signal.deflection:
            usefulness -= 15
            tags.append("self_service_deflection")
        if signal.specific and signal.context_topic and not (
            signal.proactive or signal.clarification
        ):
            usefulness -= 20
            if "specific_case_unresolved" not in tags:
                tags.append("specific_case_unresolved")

        if knowledge:
            unsupported_claims = [claim for claim in signal.claims if claim not in knowledge]
        else:
            unsupported_claims = signal.claims
        groundedness = 100.0 - min(50, 25 * len(unsupported_claims))
        if unsupported_claims:
            tags.append("unsupported_claim")
        if signal.capabilities:
            groundedness -= 15
            tags.append("unsupported_capability")

        tone = 80.0
        if signal.empathy:
            tone += 20
        if signal.emotion and not signal.empathy:
            tone -= 40
            tags.append("emotion_underaddressed")

        clarity = 90.0
        if len(reply) > 180:
            clarity -= 20
        elif len(reply) > 120:
            clarity -= 10
        if signal.actions:
            clarity += 10
        if re.search(r"加强[^\uff0c\u3002]*培训|反馈给品控|转达给产品", reply):
            clarity -= 15

        critical = bool(signal.completed_actions)
        critical_reason = (
            "声称已完成需要外部系统权限的操作，但当前没有执行证据"
            if critical
            else None
        )
        if critical:
            tags.append("unsupported_completed_action")

        tags = list(dict.fromkeys(tags))
        improvement = self._improvement(tags)
        metrics = {
            "intent_accuracy": MetricScore(
                _bounded(intent),
                "结合是否回应具体个案、必要追问和多意图覆盖评分。",
                _evidence(
                    "追问/具体个案信号",
                    signal.clarification or signal.specific,
                ),
            ),
            "usefulness": MetricScore(
                _bounded(usefulness),
                "根据主动承接、可执行步骤与自助转移信号评分。",
                _evidence(
                    "主动/转移信号",
                    signal.proactive + signal.actions + signal.deflection,
                ),
            ),
            "groundedness": MetricScore(
                _bounded(groundedness),
                (
                    "检测到缺少当前输入证据、待核实的外部声明。"
                    if unsupported_claims or signal.capabilities
                    else "未检测到需要外部证据的明确声明。"
                ),
                _evidence(
                    "待核实声明", unsupported_claims + signal.capabilities
                ),
            ),
            "tone": MetricScore(
                _bounded(tone),
                "根据用户情绪强度与回复的共情匹配程度评分。",
                _evidence("情绪/共情信号", signal.emotion + signal.empathy),
            ),
            "clarity": MetricScore(
                _bounded(clarity),
                "根据长度、步骤结构和无关内部话术评分。",
                [f"回复长度: {len(reply)} 字符"],
            ),
        }
        return CaseEvaluation(
            case_id=case["id"],
            metrics=metrics,
            risk_tags=tags,
            critical_fail=critical,
            critical_reason=critical_reason,
            improvement=improvement,
            evaluator={"mode": "mock", "version": self.version},
        )

    @staticmethod
    def _improvement(tags: list[str]) -> str:
        if "unsupported_completed_action" in tags:
            return "不要宣称已执行未经系统确认的操作。"
        if "multi_intent_missed" in tags:
            return "分别回应每个诉求，并索取各自所需的最小信息。"
        if "specific_case_unresolved" in tags:
            return "主动追问订单、账号或商品信息，针对当前个案给出下一步。"
        if "unsupported_claim" in tags:
            return "对时效、政策或商品参数提供可追溯依据，无法确认时明确说明待核实。"
        if "emotion_underaddressed" in tags:
            return "先承接用户的担忧或不满，再给出具体处理步骤。"
        return "保留直接答案，并补充更具体的下一步。"
