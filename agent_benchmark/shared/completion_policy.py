"""Deterministic completion guard shared by benchmark harnesses."""

from __future__ import annotations


def decide_completion(
    *, finish_reason: str | None, write_calls: int, test_calls: int, turn: int, max_turns: int
) -> dict[str, str]:
    """Decide whether a text-only agent response may end a coding attempt."""
    if turn >= max_turns:
        return {"action": "stop", "reason": "turn_budget_exhausted"}

    normalized = (finish_reason or "").strip().lower()
    if normalized in {"length", "max_tokens"}:
        return {"action": "continue", "reason": "output_truncated"}
    if write_calls == 0:
        return {"action": "continue", "reason": "no_write"}
    if test_calls == 0:
        return {"action": "continue", "reason": "no_test"}
    return {"action": "stop", "reason": "completed"}


def correction_prompt(reason: str) -> str:
    """Return a concise action prompt for a rejected completion attempt."""
    prompts = {
        "output_truncated": (
            "Your response was truncated. Stop explaining and use the tools now: "
            "make the smallest correct implementation change, then run public tests."
        ),
        "no_write": (
            "You have not modified an implementation file. Use write_file now to make the "
            "smallest correct fix, then run public tests. Do not finish with analysis only."
        ),
        "no_test": (
            "You changed implementation files but have not run public tests. "
            "Use run_public_tests now, fix failures if needed, then finish."
        ),
    }
    return prompts.get(reason, "Continue solving the task with tools.")
