from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator

if os.getenv("FOODCRISIS_STANDALONE") == "1":
    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        episode_id: str = ""
else:
    try:
        from openenv.core.env_server import Action, Observation, State
    except ImportError:  # pragma: no cover
        from openenv_core.env_server import Action, Observation, State

SUPPORTED_ACTIONS = {"INSPECT", "QUARANTINE", "LIFT", "RECALL", "TRACE", "ALERT", "WAIT"}


class NodeState(BaseModel):
    node_id: str
    node_type: str
    sensor_reading: float = Field(default=0.0, ge=0.0, le=1.0)
    quarantined: bool = False
    batch_ids: list[str] = Field(default_factory=list)
    connected_to: list[str] = Field(default_factory=list)


class BatchRecord(BaseModel):
    batch_id: str
    origin_node: str
    current_node: str
    path_taken: list[str] = Field(default_factory=list)
    planned_path: list[str] = Field(default_factory=list)
    contaminated: bool = False
    recalled: bool = False
    delivered: bool = False


class IllnessReport(BaseModel):
    retailer_id: str
    case_count: int = Field(default=0, ge=0)
    timestep_reported: int = Field(default=0, ge=0)
    report_text: str = ""


class PendingInspection(BaseModel):
    node_id: str
    due_timestep: int = Field(default=0, ge=0)
    contaminated: bool = False


class FoodCrisisAction(Action):
    action_type: str = "WAIT"

    @field_validator("action_type", mode="before")
    @classmethod
    def normalize_action_type(cls, value: Any) -> str:
        if value is None:
            return "WAIT"

        text = " ".join(str(value).strip().split())
        if not text:
            return "WAIT"

        parts = text.split(" ", 1)
        verb = parts[0].upper().replace("-", "_")
        target = parts[1].strip() if len(parts) > 1 else ""
        return f"{verb} {target}".strip()

    @property
    def verb(self) -> str:
        return self.action_type.split(" ", 1)[0]

    @property
    def target(self) -> str | None:
        parts = self.action_type.split(" ", 1)
        return parts[1].strip() if len(parts) > 1 else None

    @property
    def is_supported(self) -> bool:
        return self.verb in SUPPORTED_ACTIONS


class FoodCrisisObservation(Observation):
    timestep: int = Field(default=0, ge=0)
    nodes: list[NodeState] = Field(default_factory=list)
    sensor_readings: dict[str, float] = Field(default_factory=dict)
    illness_reports: list[IllnessReport] = Field(default_factory=list)
    quarantine_status: dict[str, bool] = Field(default_factory=dict)
    lab_results: dict[str, str] = Field(default_factory=dict)
    traced_batches: dict[str, list[str]] = Field(default_factory=dict)
    lab_budget: int = Field(default=0, ge=0)
    recall_budget: int = Field(default=0, ge=0)
    public_trust: float = Field(default=1.0, ge=0.0, le=1.0)
    natural_language_summary: str = ""
    goal: str = "Protect consumers by tracing contamination through the food network."
    tool_result: str = "AMBIGUOUS"
    error_type: str = "TRANSIENT"
    same_error_count: int = Field(default=0, ge=0)
    budget_remaining: float = Field(default=1.0, ge=0.0, le=1.0)
    step_count: int = Field(default=0, ge=0)
    last_action_error: bool = False
    active_tool: str = "primary"
    cooldown_remaining: int = Field(default=0, ge=0)
    progress_hint: float = Field(default=0.0, ge=0.0, le=1.0)
    history_tail: list[str] = Field(default_factory=list)
    status_summary: str = ""


class FoodCrisisState(State):
    goal: str = "Protect consumers by tracing contamination through the food network."
    task_name: str = "easy"
    timestep: int = Field(default=0, ge=0)
    nodes: dict[str, NodeState] = Field(default_factory=dict)
    true_contamination: dict[str, float] = Field(default_factory=dict)
    source_nodes: list[str] = Field(default_factory=list)
    batch_records: dict[str, BatchRecord] = Field(default_factory=dict)
    traced_batches: dict[str, list[str]] = Field(default_factory=dict)
    edge_weights: dict[str, float] = Field(default_factory=dict)
    lab_results: dict[str, str] = Field(default_factory=dict)
    pending_inspections: list[PendingInspection] = Field(default_factory=list)
    illness_reports: list[IllnessReport] = Field(default_factory=list)
    pending_illness_reports: list[IllnessReport] = Field(default_factory=list)
    quarantine_status: dict[str, bool] = Field(default_factory=dict)
    alert_timers: dict[str, int] = Field(default_factory=dict)
    lab_budget: int = Field(default=0, ge=0)
    recall_budget: int = Field(default=0, ge=0)
    public_trust: float = Field(default=1.0, ge=0.0, le=1.0)
    false_signal_nodes: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    last_action: str = "RESET"
    last_reward: float = 0.0
    last_outcome: str = "AMBIGUOUS"
    compat_error_type: str = "TRANSIENT"
    compat_same_error_count: int = Field(default=0, ge=0)
    correct_actions: int = Field(default=0, ge=0)
    total_actions: int = Field(default=0, ge=0)
    false_positive_count: int = Field(default=0, ge=0)
    contaminated_shipments: int = Field(default=0, ge=0)
    cumulative_illness_cases: int = Field(default=0, ge=0)
    exposed_contaminated_batches: int = Field(default=0, ge=0)
    total_contaminated_batches: int = Field(default=0, ge=0)
    contained_at_step: int | None = None
    batch_counter: int = Field(default=0, ge=0)
