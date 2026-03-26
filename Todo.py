"""
todo_agent.py
=============
An AI-powered todo agent that plans and solves word problems step-by-step
using OpenAI function calling with a simple in-memory todo list.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(override=True)

MODEL = "gpt-4o"
REASONING_EFFORT = "none"

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
    """
    Build and display a Rich-formatted todo report.

    Returns
    -------
    str
        The formatted report string.
    """
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
    """
    Append new todo items and return the updated report.

    Parameters
    ----------
    descriptions:
        A list of todo descriptions to add.

    Returns
    -------
    str
        The updated todo report.
    """
    todos.extend(descriptions)
    completed.extend([False] * len(descriptions))
    return get_todo_report()


def mark_complete(index: int, completion_notes: str) -> str:
    """
    Mark the todo at *index* (1-based) as complete and log notes.

    Parameters
    ----------
    index:
        1-based position of the todo to complete.
    completion_notes:
        Rich markup string describing how the todo was completed.

    Returns
    -------
    str
        The updated todo report, or an error message if *index* is invalid.
    """
    if not (1 <= index <= len(todos)):
        return f"Error: No todo at index {index}."

    completed[index - 1] = True
    console.print(completion_notes)
    return get_todo_report()


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function definitions)
# ---------------------------------------------------------------------------

CREATE_TODOS_SCHEMA: dict[str, Any] = {
    "name": "create_todos",
    "description": (
        "Add new todos from a list of descriptions and return the full list."
    ),
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
    "description": (
        "Mark complete the todo at the given 1-based position and return the full list."
    ),
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

# Map tool names to their Python implementations for dispatch
TOOL_REGISTRY: dict[str, Any] = {
    "create_todos": create_todos,
    "mark_complete": mark_complete,
}

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def handle_tool_calls(tool_calls: list[Any]) -> list[dict[str, str]]:
    """
    Execute all tool calls and return their results as OpenAI tool-role messages.

    Parameters
    ----------
    tool_calls:
        Tool call objects returned by the model.

    Returns
    -------
    list[dict[str, str]]
        A list of ``{"role": "tool", "content": ..., "tool_call_id": ...}`` dicts.
    """
    results: list[dict[str, str]] = []
    for call in tool_calls:
        tool_name: str = call.function.name
        arguments: dict[str, Any] = json.loads(call.function.arguments)

        handler = TOOL_REGISTRY.get(tool_name)
        if handler is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            result = f"Error: tool '{tool_name}' not found."
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
    """
    Drive the agentic loop until the model stops requesting tool calls.

    Parameters
    ----------
    client:
        An initialised ``OpenAI`` client.
    messages:
        Conversation history including the system and initial user message.
    """
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,  # type: ignore[arg-type]
            reasoning_effort=REASONING_EFFORT,  # type: ignore[call-arg]
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

USER_MESSAGE = """\
A train leaves Boston at 2:00 pm traveling 60 mph.
Another train leaves New York at 3:00 pm traveling 80 mph toward Boston.
When do they meet?
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Initialise the agent and run it against the sample problem."""
    client = OpenAI()

    # Reset state for a clean run
    todos.clear()
    completed.clear()

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE},
    ]

    run_agent(client, messages)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
