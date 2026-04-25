from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from irce.rewards import RewardBreakdown

if os.getenv("FOODCRISIS_STANDALONE") == "1":
    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        episode_id: str = ""
else:
    try:
        from openenv.core.env_server import Action, Observation, State
    except ImportError:  # pragma: no cover
        from openenv_core.env_server import Action, Observation, State

SUPPORTED_ACTIONS = {"INSPECT", "QUARANTINE", "LIFT", "RECALL", "TRACE", "ALERT", "WAIT", "CONCLUDE"}


class NodeState(BaseModel):
    node_id: str
    node_type: str
    sensor_reading: float = Field(default=0.0, ge=0.0, le=1.0)
    quarantined: bool = False
    batch_ids: List[str] = Field(default_factory=list)
    connected_to: List[str] = Field(default_factory=list)


class BatchRecord(BaseModel):
    batch_id: str
    origin_node: str
    current_node: str
    path_taken: List[str] = Field(default_factory=list)
    planned_path: List[str] = Field(default_factory=list)
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
    def target(self) -> Optional[str]:
        parts = self.action_type.split(" ", 1)
        return parts[1].strip() if len(parts) > 1 else None

    @property
    def is_supported(self) -> bool:
        return self.verb in SUPPORTED_ACTIONS


class AgentMemory(BaseModel):
    

    # Nodes where lab results came back "contaminated"
    confirmed_contaminated: List[str] = Field(default_factory=list)
    # Nodes where lab results came back "clean"
    confirmed_clean: List[str] = Field(default_factory=list)
    # Pending lab submissions: list of {node_id, due_timestep}
    active_pending_labs: List[Dict[str, Any]] = Field(default_factory=list)
    # For each traced batch: batch_id -> origin node
    traced_batch_origins: Dict[str, str] = Field(default_factory=dict)
    # Retailers that have received at least one illness report
    illness_retailer_ids: List[str] = Field(default_factory=list)
    # Adversarial trap nodes the agent has confirmed clean via INSPECT
    # (inspected AND lab result = "clean" AND node was a false-signal/trap node)
    identified_traps: List[str] = Field(default_factory=list)


class FoodCrisisObservation(Observation):
    timestep: int = Field(default=0, ge=0)
    nodes: List[NodeState] = Field(default_factory=list)
    sensor_readings: Dict[str, float] = Field(default_factory=dict)
    illness_reports: List[IllnessReport] = Field(default_factory=list)
    quarantine_status: Dict[str, bool] = Field(default_factory=dict)
    lab_results: Dict[str, str] = Field(default_factory=dict)
    traced_batches: Dict[str, List[str]] = Field(default_factory=dict)
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
    history_tail: List[str] = Field(default_factory=list)
    status_summary: str = ""
    reward_breakdown: Optional[RewardBreakdown] = None
    # Structured memory: everything the agent has confirmed so far
    agent_memory: Optional[AgentMemory] = None
    # Count of adversarial trap nodes in the current episode (identity is hidden from agent)
    adversarial_trap_count: int = Field(default=0, ge=0)


class FoodCrisisState(State):
    goal: str = "Protect consumers by tracing contamination through the food network."
    task_name: str = "easy"
    timestep: int = Field(default=0, ge=0)
    nodes: Dict[str, NodeState] = Field(default_factory=dict)
    true_contamination: Dict[str, float] = Field(default_factory=dict)
    source_nodes: List[str] = Field(default_factory=list)
    batch_records: Dict[str, BatchRecord] = Field(default_factory=dict)
    traced_batches: Dict[str, List[str]] = Field(default_factory=dict)
    edge_weights: Dict[str, float] = Field(default_factory=dict)
    lab_results: Dict[str, str] = Field(default_factory=dict)
    pending_inspections: List[PendingInspection] = Field(default_factory=list)
    illness_reports: List[IllnessReport] = Field(default_factory=list)
    pending_illness_reports: List[IllnessReport] = Field(default_factory=list)
    quarantine_status: Dict[str, bool] = Field(default_factory=dict)
    alert_timers: Dict[str, int] = Field(default_factory=dict)
    lab_budget: int = Field(default=0, ge=0)
    recall_budget: int = Field(default=0, ge=0)
    public_trust: float = Field(default=1.0, ge=0.0, le=1.0)
    # Renamed from false_signal_nodes — nodes injected with artificially elevated sensor readings
    adversarial_trap_nodes: List[str] = Field(default_factory=list)

    @property
    def false_signal_nodes(self) -> List[str]:
        """Deprecated alias for adversarial_trap_nodes. Use adversarial_trap_nodes instead."""
        return self.adversarial_trap_nodes

    history: List[str] = Field(default_factory=list)
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
    contained_at_step: Optional[int] = None
    batch_counter: int = Field(default=0, ge=0)
