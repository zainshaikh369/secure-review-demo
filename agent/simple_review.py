from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


REPOSITORY_ROOT = Path("/sandbox/secure-review-demo").resolve()

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}

ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
}

MAX_FILE_CHARACTERS = 10_000
MAX_REPOSITORY_CHARACTERS = 35_000


def build_repository_snapshot() -> str:
    """Read the small repository locally and build one bounded prompt."""
    sections: list[str] = []
    total_characters = 0

    for path in sorted(REPOSITORY_ROOT.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(REPOSITORY_ROOT)

        if any(part in IGNORED_DIRECTORIES for part in relative_path.parts):
            continue

        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        if len(content) > MAX_FILE_CHARACTERS:
            content = content[:MAX_FILE_CHARACTERS] + "\n[FILE TRUNCATED]"

        numbered_content = "\n".join(
            f"{line_number:>4}: {line}"
            for line_number, line in enumerate(content.splitlines(), start=1)
        )

        section = (
            f"\n===== FILE: {relative_path} =====\n"
            f"{numbered_content}\n"
        )

        remaining = MAX_REPOSITORY_CHARACTERS - total_characters
        if remaining <= 0:
            break

        if len(section) > remaining:
            sections.append(section[:remaining] + "\n[SNAPSHOT TRUNCATED]\n")
            break

        sections.append(section)
        total_characters += len(section)

    if not sections:
        raise RuntimeError("No supported repository files were found.")

    return "".join(sections)


def extract_text(content: Any) -> str:
    """Convert Gemini structured content to plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
            else:
                rendered = str(item).strip()
                if rendered:
                    parts.append(rendered)

        return "\n".join(parts)

    return str(content)


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Export it before running this script."
        )

    repository_snapshot = build_repository_snapshot()

    prompt = f"""
You are a senior Python, FastAPI, and application-security reviewer.

Review the repository snapshot below in a single response. Do not call tools,
do not request more information, and do not modify files.

Return:
1. A short repository overview.
2. Up to 10 concrete findings ranked by severity.
3. For each finding:
   - title
   - category
   - severity
   - file path and line number
   - evidence
   - impact
   - recommended fix
4. The five highest-priority actions.

Focus on:
- authentication and authorization
- hard-coded credentials or tokens
- data exposure and input validation
- FastAPI schemas, status codes, and error handling
- architecture and maintainability
- missing tests and negative test cases

Only report findings supported by the supplied code.

REPOSITORY SNAPSHOT:
{repository_snapshot}
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        max_retries=0,
    )

    try:
        response = model.invoke(prompt)
    except Exception as exc:
        message = str(exc).upper()

        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            print(
                "Gemini quota is temporarily exhausted. Wait about one minute "
                "and run the script again. This version makes only one model request.",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc

        raise

    print(extract_text(response.content))


if __name__ == "__main__":
    main()