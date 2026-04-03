from __future__ import annotations

from typing import Any

from irce.tasks import get_task_config


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def grade_containment(log: list[dict[str, Any]]) -> float:
    if not log:
        return 0.0

    final_step = log[-1]
    total = max(1, int(final_step.get("total_contaminated_batches", 1)))
    exposed = max(0, int(final_step.get("exposed_contaminated_batches", 0)))
    return _clamp01(1.0 - (exposed / total))


def grade_precision(log: list[dict[str, Any]]) -> float:
    if not log:
        return 0.0

    final_step = log[-1]
    total_actions = int(final_step.get("total_actions", 0))
    if total_actions <= 0:
        return 1.0
    correct_actions = max(0, int(final_step.get("correct_actions", 0)))
    return _clamp01(correct_actions / total_actions)


def grade_speed(log: list[dict[str, Any]]) -> float:
    if not log:
        return 0.0

    final_step = log[-1]
    contained_at_step = final_step.get("contained_at_step")
    max_steps = max(1, int(final_step.get("max_steps", len(log))))
    if contained_at_step is None:
        return 0.0
    return _clamp01(1.0 - (int(contained_at_step) / max_steps))


def grade_public_trust(log: list[dict[str, Any]]) -> float:
    if not log:
        return 0.0

    return _clamp01(log[-1].get("public_trust", 0.0))


def grade_episode(log: list[dict[str, Any]]) -> float:
    if not log:
        return 0.0

    task_id = int(log[-1].get("task_id", 1))
    weights = get_task_config(task_id).score_weights
    containment = grade_containment(log)
    precision = grade_precision(log)
    speed = grade_speed(log)
    trust = grade_public_trust(log)
    return _clamp01(
        weights.containment * containment
        + weights.precision * precision
        + weights.speed * speed
        + weights.trust * trust
    )
