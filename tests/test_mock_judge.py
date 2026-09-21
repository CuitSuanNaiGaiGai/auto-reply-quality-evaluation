import unittest

from reply_eval.mock_judge import MockJudge


class MockJudgeTests(unittest.TestCase):
    def setUp(self):
        self.judge = MockJudge()

    def evaluate(self, question, reply):
        return self.judge.evaluate(
            {"id": "case_x", "user_question": question, "auto_reply": reply}
        )

    def test_specific_case_deflection_lowers_usefulness(self):
        weak = self.evaluate(
            "我的退款什么时候到账", "请在订单详情页查看，或联系客服。"
        )
        strong = self.evaluate(
            "我的退款什么时候到账", "请提供订单号，我帮您查询退款进度。"
        )
        self.assertLess(
            weak.metrics["usefulness"].score,
            strong.metrics["usefulness"].score,
        )
        self.assertIn("self_service_deflection", weak.risk_tags)

    def test_emotional_complaint_requires_empathy(self):
        cold = self.evaluate("等了20分钟都没人理我", "请说明您的问题。")
        empathic = self.evaluate(
            "等了20分钟都没人理我",
            "非常抱歉让您久等了，请告诉我问题，我马上帮您处理。",
        )
        self.assertLess(cold.metrics["tone"].score, empathic.metrics["tone"].score)

    def test_unsupported_numeric_policy_claim_is_flagged_not_declared_false(self):
        result = self.evaluate(
            "能取消吗", "可以取消，退款会在1-3个工作日到账。"
        )
        self.assertIn("unsupported_claim", result.risk_tags)
        self.assertIn("待核实", result.metrics["groundedness"].reason)

    def test_same_input_produces_identical_result(self):
        case = {
            "id": "case_x",
            "user_question": "优惠券怎么用不了",
            "auto_reply": "请提供优惠券编号，我帮您核实。",
        }
        self.assertEqual(
            self.judge.evaluate(case).to_dict(),
            self.judge.evaluate(case).to_dict(),
        )

    def test_multi_intent_reply_is_rewarded_for_covering_both(self):
        partial = self.evaluate("退货的事顺便看看快递到没到", "请提供退货订单号。")
        complete = self.evaluate(
            "退货的事顺便看看快递到没到",
            "请提供退货订单号和快递单号，我分别帮您查询退货与物流进度。",
        )
        self.assertLess(
            partial.metrics["intent_accuracy"].score,
            complete.metrics["intent_accuracy"].score,
        )

    def test_descriptive_specific_word_is_not_treated_as_a_question(self):
        result = self.evaluate(
            "我的快递取不出来",
            "建议先查看取件码确认具体位置，如有问题请联系快递员。",
        )
        self.assertIn("missing_clarification", result.risk_tags)

    def test_future_generic_assistance_is_weaker_than_active_case_handling(self):
        weak = self.evaluate(
            "我的快递取不出来",
            "请联系快递员，如无法解决，我们会协助您处理。",
        )
        strong = self.evaluate(
            "我的快递取不出来",
            "请提供订单号，我帮您联系快递员处理。",
        )
        self.assertGreaterEqual(
            strong.metrics["usefulness"].score
            - weak.metrics["usefulness"].score,
            25,
        )

    def test_repeating_steps_to_confused_user_is_not_a_service_closure(self):
        repeated = self.evaluate(
            "退货流程太复杂了，搞半天都不知道怎么操作",
            "抱歉给您带来困扰。流程是：1.点击退货 2.选择原因 3.提交申请。",
        )
        adaptive = self.evaluate(
            "退货流程太复杂了，搞半天都不知道怎么操作",
            "抱歉让您困扰。请问您卡在哪一步？我一步步帮您处理。",
        )
        self.assertLess(repeated.metrics["usefulness"].score, 60)
        self.assertLess(
            repeated.metrics["intent_accuracy"].score,
            adaptive.metrics["intent_accuracy"].score,
        )

    def test_repeated_failure_is_not_mistaken_for_multiple_intents(self):
        result = self.evaluate(
            "上次收到是坏的，这次又是坏的",
            "非常抱歉连续出现问题，我帮您处理这次的退货。",
        )
        self.assertNotIn("multi_intent_missed", result.risk_tags)

    def test_direct_action_is_not_treated_like_pure_deflection(self):
        result = self.evaluate(
            "我的耳机才买三天就没声音",
            "抱歉给您带来不便，您可以申请退货退款或换新。",
        )
        self.assertNotIn("specific_case_unresolved", result.risk_tags)

    def test_failure_state_without_pronoun_still_needs_case_clarification(self):
        result = self.evaluate(
            "优惠券怎么用不了",
            "可能是门槛、适用范围或过期导致，请查看使用规则。",
        )
        self.assertIn("missing_clarification", result.risk_tags)

    def test_numbered_possible_causes_are_not_counted_as_action_steps(self):
        result = self.evaluate(
            "优惠券怎么用不了",
            "可能原因：1.未达门槛 2.不在范围 3.已过期。请查看使用规则。",
        )
        self.assertIn("missing_clarification", result.risk_tags)
        self.assertLess(result.metrics["usefulness"].score, 60)

    def test_new_product_troubleshooting_burden_scores_below_direct_remedy(self):
        troubleshooting = self.evaluate(
            "耳机才买三天就没声音",
            "请先尝试：1.重新配对 2.重启耳机。如仍有问题可申请换新。",
        )
        direct = self.evaluate(
            "耳机才买三天就没声音",
            "抱歉给您带来不便，可以直接申请退货退款或换新。",
        )
        self.assertLess(
            troubleshooting.metrics["usefulness"].score,
            direct.metrics["usefulness"].score,
        )


if __name__ == "__main__":
    unittest.main()
