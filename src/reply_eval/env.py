"""Minimal, non-executing dotenv support for Qwen configuration."""

from __future__ import annotations

import os
from pathlib import Path


ALLOWED_QWEN_KEYS = {"QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL"}


class EnvFileError(ValueError):
    """Raised when an explicitly selected dotenv file cannot be read safely."""


def load_qwen_env(
    explicit_path: Path | None, cwd: Path | None = None
) -> Path | None:
    """Load missing Qwen variables from a dotenv file without evaluation."""

    default_path = (cwd or Path.cwd()) / ".env"
    if explicit_path is not None and not explicit_path.exists():
        raise EnvFileError(f"env file not found: {explicit_path}")

    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    if default_path.exists() and all(
        default_path.absolute() != path.absolute() for path in candidates
    ):
        candidates.append(default_path)
    if not candidates:
        return None

    for candidate in candidates:
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise EnvFileError(
                f"cannot read env file {candidate}: {exc}"
            ) from exc
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise EnvFileError(
                    f"invalid env line {line_number} in {candidate}: "
                    "expected KEY=value"
                )
            key, value = (part.strip() for part in line.split("=", 1))
            if key not in ALLOWED_QWEN_KEYS:
                continue
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in "\"'"
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    return explicit_path or default_path
