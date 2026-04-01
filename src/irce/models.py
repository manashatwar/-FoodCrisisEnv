"""
models.py
---------
Pydantic models for the IRCE OpenEnv interface.

Three model types follow the OpenEnv spec:

    IRCEAction      — what the agent sends each step
    IRCEObservation — what the agent receives each step
    IRCEState       — hidden full state (server-side only)

All models inherit from the OpenEnv base classes (Action, Observation, State)
when running inside an OpenEnv server. In standalone mode (IRCE_STANDALONE=1),
lightweight base classes are used so inference.py and train*.py need no server.
"""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator

if os.getenv("IRCE_STANDALONE") == "1":
    class Action(BaseModel):
        """Lightweight Action base for standalone mode."""

    class Observation(BaseModel):
        """Lightweight Observation base for standalone mode."""

        done: bool = False
        reward: float = 0.0
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        """Lightweight State base for standalone mode."""

        episode_id: str = ""
else:
    try:
        from openenv.core.env_server import Action, Observation, State
    except ImportError:  # pragma: no cover
        from openenv_core.env_server import Action, Observation, State

# The five recovery actions the agent may choose at each step.
SUPPORTED_ACTIONS = {"RETRY", "MODIFY", "SWITCH", "REPLAN", "ESCALATE"}


class IRCEAction(Action):
    """
    A single recovery action chosen by the agent.

    The agent responds with exactly one of:
        RETRY    — retry the current tool (best for TRANSIENT errors)
        MODIFY   — change the request structure (best for HARD errors)
        SWITCH   — swap to backup/primary tool (best for RATE_LIMIT or loops)
        REPLAN   — reframe the approach (best for AMBIGUOUS outcomes)
        ESCALATE — hand off to a human operator (last resort only)

    Common aliases are accepted and normalised automatically
    (e.g. "SWITCH_TOOL" → "SWITCH", "MODIFY_INPUT" → "MODIFY").
    """

    action_type: str = Field(
        default="RETRY",
        description="One of RETRY, MODIFY, SWITCH, REPLAN, ESCALATE.",
    )

    @field_validator("action_type", mode="before")
    @classmethod
    def normalize_action_type(cls, value: Any) -> str:
        """Normalise and alias-resolve the action string."""
        if value is None:
            return "RETRY"

        normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        alias_map = {
            "MODIFY_INPUT": "MODIFY",
            "SWITCH_TOOL": "SWITCH",
        }
        return alias_map.get(normalized, normalized)

    @property
    def is_supported(self) -> bool:
        """Return True if action_type is one of the five supported actions."""
        return self.action_type in SUPPORTED_ACTIONS


class IRCEObservation(Observation):
    """
    The observation returned to the agent after each step.

    All fields are LLM-readable strings or scalars.
    The most important fields for action selection are error_type,
    budget_remaining, same_error_count, and cooldown_remaining.
    """

    goal: str = Field(
        default="Recover the task efficiently.",
        description="Natural-language task goal (same throughout an episode).",
    )
    tool_result: str = Field(
        default="ERROR",
        description="Last tool call outcome: SUCCESS, ERROR, or AMBIGUOUS.",
    )
    error_type: str = Field(
        default="TRANSIENT",
        description=(
            "Type of the most recent error: TRANSIENT (retry-safe), "
            "HARD (requires modification), or RATE_LIMIT (cooldown required). "
            "May be noisy on Task 2 (12%) and Task 3 (25%)."
        ),
    )
    same_error_count: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive steps with the same error type.",
    )
    budget_remaining: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Remaining API budget fraction in [0.0, 1.0]. Exhaustion ends the episode.",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken so far in this episode.",
    )
    last_action_error: bool = Field(
        default=False,
        description="True if the previous action was a bad retry (RETRY on HARD/RATE_LIMIT).",
    )
    active_tool: str = Field(
        default="primary",
        description="Which tool path is active: 'primary' or 'backup'.",
    )
    cooldown_remaining: int = Field(
        default=0,
        ge=0,
        description="Steps remaining before a rate-limited tool path is available again.",
    )
    progress_hint: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated task completion fraction in [0.0, 1.0].",
    )
    history_tail: list[str] = Field(
        default_factory=list,
        description="Short log of the last 3 (action → result) lines.",
    )
    status_summary: str = Field(
        default="",
        description="Compact LLM-readable summary of all key state fields.",
    )


class IRCEState(State):
    """
    Full hidden environment state (server-side only, not exposed to agents).

    Contains the true (non-noisy) error type, exact progress, and all
    mechanics flags that drive the transition function in environment.py.
    """

    goal: str = "Recover the task efficiently."
    task_name: str = "easy"
    current_error_type: str = Field(
        default="TRANSIENT",
        description="True error type before observation noise is applied.",
    )
    same_error_count: int = Field(default=0, ge=0)
    budget_remaining: float = Field(default=1.0, ge=0.0, le=1.0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    step_count: int = Field(default=0, ge=0)
    tool_state: str = Field(
        default="primary",
        description="Active tool path: 'primary' or 'backup'.",
    )
    replan_bonus: float = Field(default=0.0, ge=0.0, le=1.0)
    cooldown_remaining: int = Field(default=0, ge=0)
    ambiguous_count: int = Field(default=0, ge=0)
    last_tool_result: str = "ERROR"
    history: list[str] = Field(default_factory=list)
    last_action: str = "RESET"
    last_reward: float = 0.0
