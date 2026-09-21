import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositorySecurityTests(unittest.TestCase):
    def test_dotenv_is_ignored(self):
        ignored = subprocess.run(
            ["git", "check-ignore", ".env"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0)

    def test_example_contains_no_api_key(self):
        values = {}
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        self.assertEqual(values["QWEN_API_KEY"], "")

    def test_tracked_text_does_not_contain_qwen_secret_prefix(self):
        secret_pattern = re.compile(r"s" + r"k-[A-Za-z0-9._-]{12,}")
        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True
        ).splitlines()
        for relative in tracked:
            path = ROOT / relative
            if path.is_file() and path.suffix.lower() not in {".png"}:
                content = path.read_text(encoding="utf-8", errors="ignore")
                self.assertIsNone(secret_pattern.search(content), relative)


if __name__ == "__main__":
    unittest.main()
