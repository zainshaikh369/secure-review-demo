from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI


REPOSITORY_ROOT = Path("/sandbox/secure-review-demo").resolve()

IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
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

MAX_FILE_CHARACTERS = 12_000
MAX_SNAPSHOT_CHARACTERS = 45_000


def build_repository_snapshot() -> str:
    """Create one bounded repository snapshot without using model tool calls."""
    sections: list[str] = []
    total_characters = 0

    for path in sorted(REPOSITORY_ROOT.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(REPOSITORY_ROOT)

        if any(part in IGNORED_PARTS for part in relative_path.parts):
            continue

        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        if len(content) > MAX_FILE_CHARACTERS:
            content = (
                content[:MAX_FILE_CHARACTERS]
                + "\n[File truncated by repository snapshot builder]"
            )

        numbered_content = "\n".join(
            f"{line_number:>4}: {line}"
            for line_number, line in enumerate(content.splitlines(), start=1)
        )

        section = f"\n===== FILE: {relative_path} =====\n{numbered_content}\n"

        remaining = MAX_SNAPSHOT_CHARACTERS - total_characters
        if remaining <= 0:
            break

        if len(section) > remaining:
            sections.append(
                section[:remaining]
                + "\n[Repository snapshot truncated at global limit]\n"
            )
            total_characters += remaining
            break

        sections.append(section)
        total_characters += len(section)

    if not sections:
        raise RuntimeError(
            f"No supported text files were found under {REPOSITORY_ROOT}."
        )

    return "".join(sections)


def create_model() -> ChatGoogleGenerativeAI:
    """Create a no-retry model client to avoid extra quota consumption."""
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        max_retries=0,
    )


def build_agent(repository_snapshot: str):
    """
    Build a quota-efficient Deep Agents workflow.

    Expected model-call pattern:
    1. Coordinator delegates to both specialists in one turn.
    2. Security specialist responds once.
    3. Quality specialist responds once.
    4. Coordinator synthesizes once.

    Normal total: approximately four model requests.
    """
    model = create_model()

    security_prompt = f"""
You are a senior application-security reviewer.

Analyze only the repository snapshot below. You have all required evidence
already, so do not use filesystem, shell, search, todo, or other tools.

Return no more than six concrete findings. For each finding include:
- title
- severity: Critical, High, Medium, or Low
- file path and line number
- evidence
- impact
- recommended fix

Prioritize authentication, authorization, hard-coded secrets, data exposure,
input validation, injection, and insecure design. Do not speculate.

REPOSITORY SNAPSHOT:
{repository_snapshot}
"""

    quality_prompt = f"""
You are a senior Python and FastAPI software-quality reviewer.

Analyze only the repository snapshot below. You have all required evidence
already, so do not use filesystem, shell, search, todo, or other tools.

Return no more than six concrete findings covering the highest-value issues in:
- test coverage and negative tests
- API schemas and validation
- HTTP status codes and error handling
- typing and maintainability
- architecture and state management

For each finding include:
- title
- severity: Critical, High, Medium, or Low
- file path and line number
- evidence
- impact
- recommended fix

Do not repeat security findings unless they directly affect API correctness or
test coverage. Do not speculate.

REPOSITORY SNAPSHOT:
{repository_snapshot}
"""

    subagents: list[dict[str, Any]] = [
        {
            "name": "security-reviewer",
            "description": (
                "Performs one bounded security review from the supplied "
                "repository snapshot."
            ),
            "system_prompt": security_prompt,
            "tools": [],
            "model": model,
        },
        {
            "name": "quality-reviewer",
            "description": (
                "Performs one bounded FastAPI, testing, architecture, and "
                "maintainability review from the supplied repository snapshot."
            ),
            "system_prompt": quality_prompt,
            "tools": [],
            "model": model,
        },
    ]

    coordinator_prompt = """
You coordinate a quota-efficient software review.

You must follow this exact workflow:
1. In your first model turn, delegate exactly one task to security-reviewer and
   exactly one task to quality-reviewer.
2. Issue both delegations together in the same turn when possible.
3. Never call either specialist more than once.
4. Do not use filesystem, shell, search, todo, or file-writing tools.
5. After both specialist reports return, produce the final report immediately.
6. Do not request additional analysis.

The final report must contain:
- repository overview
- specialists used
- consolidated findings ranked by severity and practical impact
- duplicate findings merged
- exact file and line references
- five highest-priority remediation actions

Keep the final report focused and evidence-based.
"""

    return create_deep_agent(
        model=model,
        tools=[],
        subagents=subagents,
        system_prompt=coordinator_prompt,
    )


def extract_text(content: Any) -> str:
    """Convert structured Gemini message content into readable text."""
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


def is_quota_error(error: Exception) -> bool:
    """Recognize Gemini rate-limit and quota errors without SDK coupling."""
    message = str(error).upper()
    return (
        "RESOURCE_EXHAUSTED" in message
        or "429" in message
        or "QUOTA" in message
    )


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Export it before running the agent."
        )

    snapshot = build_repository_snapshot()
    agent = build_agent(snapshot)

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Run the bounded repository review now. Delegate "
                            "once to each configured specialist, then synthesize "
                            "the final report without additional tool use."
                        ),
                    }
                ]
            },
            config={"recursion_limit": 12},
        )
    except Exception as exc:
        if is_quota_error(exc):
            print(
                "Gemini quota was exceeded. Wait for the rate-limit window "
                "to reset, then rerun this command. This optimized workflow "
                "normally uses about four model requests.",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        raise

    final_message = result["messages"][-1]
    print(extract_text(final_message.content))


if __name__ == "__main__":
    main()