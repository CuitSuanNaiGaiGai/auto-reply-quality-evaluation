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
        payload = [
            {
                "id": "case_02",
                "human_reference": "r",
                "annotator_notes": "n",
            }
        ]
        with self.assertRaisesRegex(InputError, "ID mismatch"):
            load_human_references(self.write_json(payload), {"case_01"})

    def test_valid_cases_are_loaded(self):
        payload = [{"id": "case_01", "user_question": "q", "auto_reply": "a"}]
        self.assertEqual(load_cases(self.write_json(payload)), payload)


if __name__ == "__main__":
    unittest.main()
