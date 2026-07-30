from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool


REPOSITORY_ROOT = Path("/sandbox/secure-review-demo").resolve()


@tool
def read_repository_file(relative_path: str) -> str:
    """Read a UTF-8 text file from the approved repository."""
    target = (REPOSITORY_ROOT / relative_path).resolve()

    if target != REPOSITORY_ROOT and REPOSITORY_ROOT not in target.parents:
        raise ValueError("Access outside the repository is denied.")

    if not target.is_file():
        raise FileNotFoundError(f"File not found: {relative_path}")

    return target.read_text(encoding="utf-8")


agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[read_repository_file],
    system_prompt=(
        "You are a secure repository-review agent. "
        "Use only the provided repository tools. "
        "Never access files outside the approved repository."
    ),
)


def main() -> None:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Read README.md and return only its first heading.",
                }
            ]
        }
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()