"""
tasks.py
--------
Three deterministic task configurations for the IRCE benchmark.

Difficulty progression
----------------------
Task 1 — easy
    Clean signals, generous budget, no rate limits, no drift.
    Tests basic repair decisions without overreacting.

Task 2 — medium
    Adds RATE_LIMIT errors, cooldown mechanics, and moderate ambiguity.
    Tests routing around degraded tool paths.

Task 3 — hard
    Lower budget, noisy error labels, failure-mode drift after step 2,
    cascading repeat penalties, and costly backup routing.
    Tests stability under uncertainty while finishing efficiently.

All tasks are seeded (deterministic) and produce scores in [0.0, 1.0].
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskConfig:
    """
    Immutable configuration for one IRCE task.

    Probability fields (retry_success, modify_success, etc.) control the
    chance that a given action succeeds under its ideal error condition.
    The backup_success_bonus is added when the agent is on the backup tool path.
    """

    task_id: int
    name: str
    description: str
    goal: str
    initial_budget: float       # Starting budget in [0.0, 1.0]
    max_steps: int              # Hard episode step limit
    noise_level: float          # Probability that observed error_type is wrong
    ambiguity_rate: float       # Base chance an action returns AMBIGUOUS
    allow_rate_limit: bool      # Whether RATE_LIMIT errors can occur
    rate_limit_cooldown: int    # Steps of cooldown imposed by RATE_LIMIT
    drift_after_step: int | None  # Step after which failure modes can escalate
    drift_enabled: bool         # Master flag for drift mechanics
    cascade_penalty: float      # Extra penalty per repeated failure (hard task)
    initial_errors: tuple[str, ...]  # Error types the episode can start with
    retry_success: float        # P(success | RETRY, TRANSIENT, primary)
    modify_success: float       # P(success | MODIFY, HARD)
    switch_success: float       # P(success | SWITCH, RATE_LIMIT)
    replan_success: float       # P(success | REPLAN, HARD)
    backup_success_bonus: float # Added to success probability on backup path
    backup_budget_cost: float   # Extra budget consumed per step on backup
    ambiguity_progress: float   # Progress fraction awarded on AMBIGUOUS result


def build_task_registry() -> dict[int, TaskConfig]:
    """Return a fresh mapping of task_id → TaskConfig for all three tasks."""
    return {
        1: TaskConfig(
            task_id=1,
            name="easy",
            description="Single-workflow recovery with transient and hard failures, generous budget, and clean signals.",
            goal="Recover a stalled production workflow with minimal wasted attempts.",
            initial_budget=1.0,
            max_steps=6,
            noise_level=0.0,
            ambiguity_rate=0.05,
            allow_rate_limit=False,
            rate_limit_cooldown=0,
            drift_after_step=None,
            drift_enabled=False,
            cascade_penalty=0.0,
            initial_errors=("TRANSIENT", "HARD"),
            retry_success=0.68,
            modify_success=0.76,
            switch_success=0.52,
            replan_success=0.45,
            backup_success_bonus=0.12,
            backup_budget_cost=0.03,
            ambiguity_progress=0.15,
        ),
        2: TaskConfig(
            task_id=2,
            name="medium",
            description="Two-tool recovery with rate limits, moderate ambiguity, and noticeable budget pressure.",
            goal="Recover a mixed-failure tool workflow without wasting API budget.",
            initial_budget=0.8,
            max_steps=6,
            noise_level=0.12,
            ambiguity_rate=0.12,
            allow_rate_limit=True,
            rate_limit_cooldown=1,
            drift_after_step=None,
            drift_enabled=False,
            cascade_penalty=0.04,
            initial_errors=("TRANSIENT", "HARD", "RATE_LIMIT"),
            retry_success=0.58,
            modify_success=0.66,
            switch_success=0.64,
            replan_success=0.45,
            backup_success_bonus=0.16,
            backup_budget_cost=0.06,
            ambiguity_progress=0.18,
        ),
        3: TaskConfig(
            task_id=3,
            name="hard",
            description="Drifting reliability, noisy error labels, cooldown pressure, and costly backup routing.",
            goal="Recover an unstable workflow under drift, ambiguity, and tight budget limits.",
            initial_budget=0.6,
            max_steps=7,
            noise_level=0.25,
            ambiguity_rate=0.2,
            allow_rate_limit=True,
            rate_limit_cooldown=2,
            drift_after_step=2,
            drift_enabled=True,
            cascade_penalty=0.08,
            initial_errors=("TRANSIENT", "HARD", "RATE_LIMIT"),
            retry_success=0.5,
            modify_success=0.62,
            switch_success=0.62,
            replan_success=0.56,
            backup_success_bonus=0.18,
            backup_budget_cost=0.08,
            ambiguity_progress=0.2,
        ),
    }


TASKS = build_task_registry()


def get_task_config(task_id: int = 1) -> TaskConfig:
    """
    Return the TaskConfig for the given task_id (1, 2, or 3).

    Raises ValueError for unknown task IDs.
    """
    try:
        return TASKS[int(task_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported task_id: {task_id}") from exc
