"""
todo_agent.py
=============
An AI-powered todo agent that plans and solves problems step-by-step
using OpenRouter's free model tier with function/tool calling.

Setup
-----
1. Create a free account at https://openrouter.ai and generate an API key.
2. Add the key to your .env file:

    OPENROUTER_API_KEY=sk-or-v1-...

3. Install dependencies:

    pip install openai python-dotenv rich

Usage
-----
    python todo_agent.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.prompt import Prompt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# "openrouter/free" auto-selects the best available free model and smartly
# filters for models that support the features your request needs (tool calling).
# This means no hardcoded model names that can get decommissioned.
MODEL = "openrouter/free"

# Optional: shown in OpenRouter's analytics dashboard.
APP_TITLE = "Todo Agent"
APP_SITE_URL = "https://github.com/your-username/todo-agent"

console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

todos: list[str] = []
completed: list[bool] = []

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def show(text: str) -> None:
    """Render *text* via Rich console, falling back to plain print on error."""
    try:
        console.print(text)
    except Exception:  # noqa: BLE001
        print(text)


# ---------------------------------------------------------------------------
# Todo operations
# ---------------------------------------------------------------------------


def get_todo_report() -> str:
    """Build and display a Rich-formatted todo report."""
    lines: list[str] = []
    for index, (todo, done) in enumerate(zip(todos, completed), start=1):
        if done:
            lines.append(f"Todo #{index}: [green][strike]{todo}[/strike][/green]")
        else:
            lines.append(f"Todo #{index}: {todo}")

    report = "\n".join(lines)
    show(report)
    return report


def create_todos(descriptions: list[str]) -> str:
    """Append new todo items and return the updated report."""
    todos.extend(descriptions)
    completed.extend([False] * len(descriptions))
    return get_todo_report()


def mark_complete(index: int, completion_notes: str) -> str:
    """Mark the todo at *index* (1-based) as complete and log notes."""
    if not (1 <= index <= len(todos)):
        return f"Error: No todo at index {index}."

    completed[index - 1] = True
    console.print(completion_notes)
    return get_todo_report()


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

CREATE_TODOS_SCHEMA: dict[str, Any] = {
    "name": "create_todos",
    "description": "Add new todos from a list of descriptions and return the full list.",
    "parameters": {
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "title": "Descriptions",
                "description": "List of todo item descriptions to create.",
            }
        },
        "required": ["descriptions"],
        "additionalProperties": False,
    },
}

MARK_COMPLETE_SCHEMA: dict[str, Any] = {
    "name": "mark_complete",
    "description": "Mark complete the todo at the given 1-based position and return the full list.",
    "parameters": {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "title": "Index",
                "description": "The 1-based index of the todo to mark as complete.",
            },
            "completion_notes": {
                "type": "string",
                "title": "Completion Notes",
                "description": "Notes about how the todo was completed (Rich markup supported).",
            },
        },
        "required": ["index", "completion_notes"],
        "additionalProperties": False,
    },
}

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": CREATE_TODOS_SCHEMA},
    {"type": "function", "function": MARK_COMPLETE_SCHEMA},
]

TOOL_REGISTRY: dict[str, Any] = {
    "create_todos": create_todos,
    "mark_complete": mark_complete,
}

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def handle_tool_calls(tool_calls: list[Any]) -> list[dict[str, str]]:
    """Execute all tool calls and return their results as tool-role messages."""
    results: list[dict[str, str]] = []
    for call in tool_calls:
        tool_name: str = call.function.name
        arguments: dict[str, Any] = json.loads(call.function.arguments)

        handler = TOOL_REGISTRY.get(tool_name)
        if handler is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            result: Any = f"Error: tool '{tool_name}' not found."
        else:
            result = handler(**arguments)

        results.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": call.id,
            }
        )
    return results


def run_agent(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
) -> None:
    """Drive the agentic loop until the model stops requesting tool calls."""
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,  # type: ignore[arg-type]
            extra_headers={
                "HTTP-Referer": APP_SITE_URL,
                "X-Title": APP_TITLE,
            },
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            message = choice.message
            tool_results = handle_tool_calls(message.tool_calls)  # type: ignore[arg-type]
            messages.append(message)  # type: ignore[arg-type]
            messages.extend(tool_results)  # type: ignore[arg-type]
        else:
            show(choice.message.content or "")
            break


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are given a problem to solve. Use your todo tools to:
1. Plan a list of steps.
2. Carry out each step in turn.

Rules:
- If any quantity is not provided, include a step to estimate it reasonably.
- Provide your final solution in Rich console markup (no code blocks).
- Do not ask the user for clarification; respond only with the answer after using your tools.
"""

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def build_client() -> OpenAI:
    """
    Construct an OpenAI client configured to use OpenRouter's API.

    Returns
    -------
    OpenAI
        A configured client instance.

    Raises
    ------
    EnvironmentError
        If ``OPENROUTER_API_KEY`` is not set in the environment.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set.\n"
            "  1. Create a free account at https://openrouter.ai\n"
            "  2. Generate an API key\n"
            "  3. Add  OPENROUTER_API_KEY=sk-or-v1-...  to your .env file"
        )

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Ask the user for a question, then run the agent to solve it."""
    client = build_client()

    console.print("\n[bold cyan]🤖 AI Todo Agent (powered by OpenRouter)[/bold cyan]")
    console.print("[dim]Type your question and press Enter. Type 'exit' to quit.[/dim]\n")

    while True:
        user_question = Prompt.ask("[bold yellow]Your Question[/bold yellow]").strip()

        if not user_question:
            console.print("[red]Please enter a question.[/red]\n")
            continue

        if user_question.lower() in {"exit", "quit", "q"}:
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break

        todos.clear()
        completed.clear()

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ]

        console.print("\n[dim]🧠 Thinking...[/dim]\n")
        run_agent(client, messages)
        console.print("\n" + "─" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
