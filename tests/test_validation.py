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

    def test_ties_receive_average_rank(self):
        self.assertEqual(spearman_rank([1, 1, 2], [10, 10, 20]), 1.0)


if __name__ == "__main__":
    unittest.main()
