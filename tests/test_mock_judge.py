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


if __name__ == "__main__":
    unittest.main()
