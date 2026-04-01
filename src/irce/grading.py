"""
grading.py
----------
Deterministic episode grader for IRCE.

Final score formula (returns a float in [0.0, 1.0]):

    score = 0.45 × completion
          + 0.25 × efficiency
          + 0.15 × cost
          + 0.15 × recovery_quality

Component definitions
---------------------
completion (0–1)
    1.0 if the task reached 100% progress.
    0.4 if the agent escalated after ≥ 50% progress (controlled exit).
    0.0 otherwise.

efficiency (0–1)
    1 - (steps_used - 1) / max_steps.
    Fewer steps → higher score.

cost (0–1)
    budget_remaining at episode end.
    Preserving budget is rewarded.

recovery_quality (0–1)
    Penalises bad retries on HARD/RATE_LIMIT errors and budget exhaustion.
    Rewards resolving ambiguous outcomes and controlled escalation.
"""
from __future__ import annotations

from typing import Any


def _clamp01(value: float) -> float:
    """Clamp value to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def grade_completion(log: list[dict[str, Any]]) -> float:
    """
    Score task completion.

    Returns 1.0 on full completion, 0.4 for a controlled escalation after
    meaningful progress (≥ 50%), and 0.0 for all other outcomes.
    """
    if not log:
        return 0.0

    final_step = log[-1]
    if final_step.get("completed", False):
        return 1.0

    if final_step.get("action") == "ESCALATE" and final_step.get("progress", 0.0) >= 0.5:
        return 0.4

    return 0.0


def grade_efficiency(log: list[dict[str, Any]]) -> float:
    """
    Score step efficiency.

    Computed as 1 - (steps_used - 1) / max_steps, clamped to [0, 1].
    A one-step solve gives the maximum; using all steps gives the minimum.
    """
    if not log:
        return 0.0

    max_steps = max(1, int(log[-1].get("max_steps", len(log))))
    steps = max(1, int(log[-1].get("step_count", len(log))))
    return _clamp01(1.0 - ((steps - 1) / max_steps))


def grade_cost(log: list[dict[str, Any]]) -> float:
    """
    Score budget preservation.

    Returns budget_remaining at episode end, clamped to [0, 1].
    """
    if not log:
        return 0.0

    return _clamp01(log[-1].get("budget_remaining", 0.0))


def grade_recovery_quality(log: list[dict[str, Any]]) -> float:
    """
    Score recovery behaviour quality.

    Penalises:
        - Bad retries (RETRY on HARD/RATE_LIMIT) — −0.45 × fraction of steps
        - Budget exhaustion — −0.2 flat

    Rewards:
        - Resolving ambiguous outcomes — +0.15 × resolution rate
        - Controlled escalation (escalate with budget remaining) — +0.05
    """
    if not log:
        return 0.0

    steps = max(1, len(log))
    bad_retries = sum(1 for step in log if step.get("bad_retry"))
    ambiguous_steps = sum(1 for step in log if step.get("tool_result") == "AMBIGUOUS")
    resolved_ambiguity = sum(1 for step in log if step.get("resolved_ambiguity"))
    exhausted_budget = 1 if log[-1].get("budget_remaining", 0.0) <= 0.0 else 0
    controlled_escalation = (
        1
        if log[-1].get("action") == "ESCALATE" and log[-1].get("budget_remaining", 0.0) > 0.0
        else 0
    )

    ambiguity_score = 1.0
    if ambiguous_steps:
        ambiguity_score = resolved_ambiguity / ambiguous_steps

    quality = (
        1.0
        - 0.45 * (bad_retries / steps)
        - 0.2 * exhausted_budget
        + 0.15 * ambiguity_score
        + 0.05 * controlled_escalation
    )
    return _clamp01(quality)


def grade_episode(log: list[dict[str, Any]]) -> float:
    """
    Compute the final IRCE episode score in [0.0, 1.0].

    Applies the four-component weighted formula:
        0.45 × completion + 0.25 × efficiency + 0.15 × cost + 0.15 × recovery_quality
    """
    completion = grade_completion(log)
    efficiency = grade_efficiency(log)
    cost = grade_cost(log)
    recovery_quality = grade_recovery_quality(log)
    return _clamp01(
        0.45 * completion
        + 0.25 * efficiency
        + 0.15 * cost
        + 0.15 * recovery_quality
    )
