import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reply_eval.env import EnvFileError, load_qwen_env


class EnvTests(unittest.TestCase):
    def write_env(self, content: str, filename: str = ".env") -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_explicit_file_loads_only_qwen_keys_without_expansion(self):
        path = self.write_env(
            'QWEN_API_KEY="file-key"\n'
            "QWEN_MODEL='qwen-plus'\n"
            "OTHER=blocked\n"
            'QWEN_BASE_URL="https://example.test/$TOKEN"\n'
        )
        with patch.dict(os.environ, {}, clear=True):
            loaded = load_qwen_env(path, path.parent)
            self.assertEqual(loaded, path)
            self.assertEqual(os.environ["QWEN_API_KEY"], "file-key")
            self.assertEqual(os.environ["QWEN_MODEL"], "qwen-plus")
            self.assertEqual(
                os.environ["QWEN_BASE_URL"], "https://example.test/$TOKEN"
            )
            self.assertNotIn("OTHER", os.environ)

    def test_shell_values_override_explicit_file(self):
        path = self.write_env(
            "QWEN_API_KEY=file-key\nQWEN_MODEL=file-model\n"
        )
        with patch.dict(
            os.environ,
            {"QWEN_API_KEY": "shell-key", "QWEN_MODEL": "shell-model"},
            clear=True,
        ):
            load_qwen_env(path, path.parent)
            self.assertEqual(os.environ["QWEN_API_KEY"], "shell-key")
            self.assertEqual(os.environ["QWEN_MODEL"], "shell-model")

    def test_explicit_file_falls_through_to_default_for_missing_keys(self):
        explicit = self.write_env(
            "QWEN_MODEL=explicit-model\n", filename="custom.env"
        )
        default = self.write_env(
            "QWEN_MODEL=default-model\n"
            "QWEN_BASE_URL=https://default.test/v1\n"
        )
        with patch.dict(
            os.environ, {"QWEN_API_KEY": "shell-key"}, clear=True
        ):
            loaded = load_qwen_env(explicit, default.parent)
            self.assertEqual(loaded, explicit)
            self.assertEqual(os.environ["QWEN_API_KEY"], "shell-key")
            self.assertEqual(os.environ["QWEN_MODEL"], "explicit-model")
            self.assertEqual(
                os.environ["QWEN_BASE_URL"], "https://default.test/v1"
            )

    def test_default_file_is_loaded_from_requested_directory(self):
        path = self.write_env("QWEN_API_KEY=default-key\n")
        with patch.dict(os.environ, {}, clear=True):
            loaded = load_qwen_env(None, path.parent)
            self.assertEqual(loaded, path)
            self.assertEqual(os.environ["QWEN_API_KEY"], "default-key")

    def test_missing_explicit_file_is_an_error(self):
        missing = Path(tempfile.gettempdir()) / "missing-reply-eval.env"
        with self.assertRaisesRegex(EnvFileError, "env file not found"):
            load_qwen_env(missing)

    def test_invalid_assignment_reports_line_number(self):
        path = self.write_env("QWEN_API_KEY=file-key\nnot-an-assignment\n")
        with self.assertRaisesRegex(EnvFileError, "line 2"):
            load_qwen_env(path)


if __name__ == "__main__":
    unittest.main()
