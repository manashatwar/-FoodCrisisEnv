"""
rewards.py
----------
Per-step reward function for IRCE.

compute_step_reward() is called by the environment after each action.
It returns a RewardBreakdown namedtuple with every component broken out,
making reward shaping transparent and auditable.

Reward signal components
------------------------
step_cost           : −0.10  always (cost of one step)
progress_bonus      : +0.30  if progress_delta ≥ 45% (significant advance)
ambiguity_bonus     : +0.15  if result=AMBIGUOUS but progress still moved
completion_bonus    : +1.00  on task completion
repeated_error_pen  : −0.20  per re-occurrence of the same error type
bad_retry_penalty   : −0.30  if RETRY is used on a HARD or RATE_LIMIT error
switch_cost_penalty : −0.05  on tool switch (−small for backup budget drain)
escalation_penalty  : −0.15  for premature escalation (low progress)
cascade_penalty     : −k×n   task-3 extra penalty for repeated failures
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    step_cost: float
    progress_bonus: float
    ambiguity_bonus: float
    completion_bonus: float
    repeated_error_penalty: float
    bad_retry_penalty: float
    switch_cost_penalty: float
    escalation_penalty: float
    cascade_penalty: float
    total: float


def compute_step_reward(
    *,
    action_type: str,
    tool_result: str,
    previous_error_type: str,
    same_error_count: int,
    progress_delta: float,
    completed: bool,
    switched_tools: bool,
    backup_budget_cost: float,
    cascade_penalty: float,
    escalated_early: bool,
) -> RewardBreakdown:
    step_cost = -0.1
    progress_bonus = 0.3 if progress_delta >= 0.45 else 0.0
    ambiguity_bonus = 0.15 if tool_result == "AMBIGUOUS" and progress_delta > 0.0 else 0.0
    completion_bonus = 1.0 if completed else 0.0
    repeated_error_penalty = -0.2 if tool_result == "ERROR" and same_error_count > 0 else 0.0
    bad_retry_penalty = (
        -0.3
        if action_type == "RETRY" and previous_error_type in {"HARD", "RATE_LIMIT"}
        else 0.0
    )
    switch_cost_penalty = -0.05 if switched_tools else -0.02 * backup_budget_cost
    escalation_penalty = -0.15 if escalated_early else 0.0
    cascade_penalty_value = -cascade_penalty * min(same_error_count, 2) if tool_result == "ERROR" else 0.0

    total = (
        step_cost
        + progress_bonus
        + ambiguity_bonus
        + completion_bonus
        + repeated_error_penalty
        + bad_retry_penalty
        + switch_cost_penalty
        + escalation_penalty
        + cascade_penalty_value
    )

    return RewardBreakdown(
        step_cost=step_cost,
        progress_bonus=progress_bonus,
        ambiguity_bonus=ambiguity_bonus,
        completion_bonus=completion_bonus,
        repeated_error_penalty=repeated_error_penalty,
        bad_retry_penalty=bad_retry_penalty,
        switch_cost_penalty=switch_cost_penalty,
        escalation_penalty=escalation_penalty,
        cascade_penalty=cascade_penalty_value,
        total=round(total, 3),
    )
